"""Browse manifest images and training preprocessing locally."""

import base64
import html
import io
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch
from PIL import Image
from pydantic import Field
from torchvision.transforms.functional import to_pil_image

from computer_vision.config import ConfigModel, parse_and_validate
from computer_vision.data import DatasetConfig, ImageDataModule, ManifestDataset
from computer_vision.evaluate import load_checkpoint
from computer_vision.lightning_module import ClassificationModule


class ViewerConfig(ConfigModel):
    """Typed dataset settings and local server options."""

    dataset: DatasetConfig
    port: int = Field(default=8080, ge=1, le=65535)
    seed: int = Field(default=42, ge=0)
    checkpoint: Path | None = None
    """Optional training checkpoint used for CPU softmax scores."""


def load_viewer_checkpoint(config: ViewerConfig) -> tuple[ViewerConfig, ClassificationModule | None]:
    """Load a model once and use its saved preprocessing and fold settings."""
    if config.checkpoint is None:
        return config, None
    model, training_config = load_checkpoint(config.checkpoint)
    model = model.cpu().eval()
    dataset = training_config.dataset.model_copy(update={"kind": "manifest", "root": config.dataset.root})
    return config.model_copy(update={"dataset": dataset}), model


def image_uri(image: Image.Image) -> str:
    """Encode a preview without exposing a file-serving endpoint."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_page(config: ViewerConfig, query: dict[str, list[str]], model: ClassificationModule | None = None) -> str:
    """Render filters, metadata, and previews from the training dataset."""
    if config.dataset.kind != "manifest":
        raise ValueError("Viewer requires dataset.kind=manifest")
    split = query.get("split", ["train"])[0]
    if split not in {"train", "validation", "test"}:
        raise ValueError("Split must be train, validation, or test")
    dataset = ManifestDataset(config.dataset, test=split == "test")
    if model is not None and model.class_names is not None and tuple(dataset.classes) != model.class_names:
        raise ValueError("Viewer manifest classes must match the checkpoint class ordering")
    module = ImageDataModule(config.dataset)
    if split == "test":
        split_indices = list(range(len(dataset)))
    else:
        module.setup("fit")
        subset = module.train_dataset if split == "train" else module.validation_dataset
        split_indices = subset.indices
    source = query.get("source", [""])[0]
    label = query.get("label", [""])[0]
    indices = [
        index
        for index in split_indices
        if (not source or dataset.records[index].source == source) and (not label or dataset.records[index].label == label)
    ]
    selectors = []
    for name, choices, selected in (
        ("split", ["train", "validation", "test"], split),
        ("source", ["", *sorted({record.source for record in dataset.records})], source),
        ("label", ["", *dataset.classes], label),
    ):
        options = "".join(
            f'<option value="{html.escape(choice, quote=True)}" {"selected" if choice == selected else ""}>{html.escape(choice or "All")}</option>'
            for choice in choices
        )
        selectors.append(f'<label>{name} <select name="{name}">{options}</select></label>')
    position = max(0, min(int(query.get("position", ["0"])[0]), len(indices) - 1))
    body = "<p>No images match these filters.</p>"
    if indices:
        index = indices[position]
        record = dataset.records[index]
        original = dataset.original_image(index)
        model_input, class_index = dataset[index]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed + index)
            augmented = module.augmentation(model_input.unsqueeze(0))[0]
        previews = "".join(
            f'<figure><figcaption>{title}</figcaption><img src="{image_uri(image)}" alt="{title}"></figure>'
            for title, image in (
                ("Original", original),
                ("Model input", to_pil_image(model_input)),
                ("Augmentation preview", to_pil_image(augmented)),
            )
        )
        metadata = html.escape(record.model_dump_json(indent=2))
        prediction = ""
        if model is not None:
            with torch.inference_mode():
                scores = model(model_input.unsqueeze(0)).softmax(dim=1)[0]
            class_names = model.class_names or tuple(f"Class {index}" for index in range(len(scores)))
            score_rows = "".join(
                f"<tr><td>{html.escape(name)}</td><td>{float(score):.6f}</td></tr>" for name, score in zip(class_names, scores, strict=True)
            )
            predicted_class = html.escape(class_names[int(scores.argmax())])
            prediction = (
                f"<h2>CPU prediction</h2><p>Predicted class: {predicted_class}.</p>"
                f"<table><thead><tr><th>Class</th><th>Softmax score</th></tr></thead><tbody>{score_rows}</tbody></table>"
            )
            if model.class_names is None:
                prediction += "<p>This checkpoint has no saved class names. Scores use output indices.</p>"
        body = (
            f"<p>Sample {position + 1} of {len(indices)}; class index {class_index}; input range [0, 1].</p>"
            f'<div class="images">{previews}</div>{prediction}<pre>{metadata}</pre>'
        )
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Training image viewer</title>
<style>body{{font:16px sans-serif;margin:2rem;max-width:1100px}}form{{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}}
.images{{display:flex;flex-wrap:wrap}}figure{{margin:1rem}}img{{width:240px;image-rendering:pixelated}}pre{{white-space:pre-wrap}}</style>
<h1>Training image viewer</h1><form>{"".join(selectors)}<label>Position (starts at 0)
<input name="position" type="number" min="0" max="{max(0, len(indices) - 1)}" value="{position}"></label>
<button>Show</button><button name="position" value="{max(0, position - 1)}" onclick="this.form.querySelector('input').disabled=true">Previous</button>
<button name="position" value="{min(len(indices) - 1, position + 1)}" onclick="this.form.querySelector('input').disabled=true">Next</button>
</form><p>Validation fold {config.dataset.validation_fold} of {config.dataset.num_folds};
fold strategy: {config.dataset.fold_strategy}; model input: {config.dataset.image_size} x {config.dataset.image_size}.</p>
{body}<details><summary>Preprocessing and augmentation settings</summary>
<pre>{html.escape(config.dataset.model_dump_json(indent=2))}</pre></details></html>'''


def main() -> None:
    """Serve the viewer on the loopback interface."""
    logging.basicConfig(level=logging.INFO)
    config, model = load_viewer_checkpoint(parse_and_validate(ViewerConfig))
    render_page(config, {}, model)

    class Handler(BaseHTTPRequestHandler):
        """Serve only image previews selected from the configured manifest."""

        def do_GET(self) -> None:
            """Return the selected sample page."""
            request = urlsplit(self.path)
            if request.path != "/":
                self.send_error(404)
                return
            try:
                page = render_page(config, parse_qs(request.query), model).encode()
            except (ValueError, OSError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - standard HTTP server argument.
            """Send request messages to the project logger."""
            logging.getLogger(__name__).info(format, *args)

    logging.getLogger(__name__).info("Viewer: http://127.0.0.1:%d", config.port)
    with HTTPServer(("127.0.0.1", config.port), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Viewer stopped")


if __name__ == "__main__":
    main()

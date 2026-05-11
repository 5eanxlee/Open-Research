from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

from open_research.core.config import Settings
from open_research.core.domain import PromptMode


class PromptTemplateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResolvedPromptTemplate:
    role: str
    source: str
    path: str | None
    template_version: str
    prompt_mode: PromptMode
    model_family: str
    rendered_body: str


class PromptTemplateLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[tuple[str, float], str] = {}

    def resolve(
        self,
        *,
        role: str,
        model_family: str,
        template_version: str,
        variables: dict[str, Any],
        fallback_body: str,
    ) -> ResolvedPromptTemplate:
        prompt_mode = PromptMode(self.settings.prompt_mode)
        if prompt_mode == PromptMode.CODE or not self.settings.prompt_externalization_enabled:
            return ResolvedPromptTemplate(
                role=role,
                source="code",
                path=None,
                template_version=template_version,
                prompt_mode=prompt_mode,
                model_family=model_family,
                rendered_body=fallback_body,
            )

        template_path = self._resolve_template_path(role=role, model_family=model_family)
        if template_path is None:
            if prompt_mode == PromptMode.TEMPLATE or self.settings.prompt_template_strict:
                raise PromptTemplateError(
                    f"No prompt template found for role={role} family={model_family}."
                )
            return ResolvedPromptTemplate(
                role=role,
                source="code",
                path=None,
                template_version=template_version,
                prompt_mode=prompt_mode,
                model_family=model_family,
                rendered_body=fallback_body,
            )

        rendered = self._render_template(template_path, variables)
        return ResolvedPromptTemplate(
            role=role,
            source="template",
            path=str(template_path),
            template_version=template_version,
            prompt_mode=prompt_mode,
            model_family=model_family,
            rendered_body=rendered,
        )

    def _resolve_template_path(self, *, role: str, model_family: str) -> Path | None:
        root = (
            Path(self.settings.prompt_template_root)
            if self.settings.prompt_template_root
            else Path(__file__).resolve().parent / "prompts"
        )
        family_path = root / role / f"{model_family}.md.j2"
        default_path = root / role / "default.md.j2"
        if family_path.exists():
            return family_path
        if self.settings.prompt_template_family_fallback and default_path.exists():
            return default_path
        return None

    def _render_template(self, path: Path, variables: dict[str, Any]) -> str:
        template = self._load_template(path)
        field_names = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name is not None and field_name != ""
        }
        missing = sorted(name for name in field_names if name not in variables)
        if missing:
            raise PromptTemplateError(
                f"Prompt template {path} is missing required variables: {', '.join(missing)}"
            )
        try:
            return template.format_map(variables)
        except KeyError as exc:  # pragma: no cover - guarded above
            raise PromptTemplateError(f"Failed to render {path}: missing {exc}") from exc

    def _load_template(self, path: Path) -> str:
        stat = path.stat()
        cache_key = (str(path), stat.st_mtime)
        if self.settings.prompt_reload_in_dev or self.settings.environment == "development":
            self._cache = {key: value for key, value in self._cache.items() if key == cache_key}
        if cache_key not in self._cache:
            self._cache[cache_key] = path.read_text(encoding="utf-8")
        return self._cache[cache_key]

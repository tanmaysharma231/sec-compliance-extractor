from __future__ import annotations

from .module import RuleExtractorModule
from .schemas import RuleExtractorInput, RuleExtractorOutput

try:
    from fastapi import FastAPI
except Exception:
    FastAPI = None

if FastAPI is not None:
    app = FastAPI(title="SEC Rule Extractor", version="0.2.0")
    module = RuleExtractorModule()

    @app.post("/interpret", response_model=RuleExtractorOutput)
    def interpret(payload: RuleExtractorInput) -> RuleExtractorOutput:
        return module.run(payload)
else:
    app = None

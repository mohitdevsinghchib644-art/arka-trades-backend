from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import json

from models.screener import (
    SetupCreateRequest,
    SetupResponse,
    ScanRequest,
    ScanResponse,
)
from database import load_setups_from_db, save_setup_to_db, delete_setup_from_db
from services.scan_engine import (
    RULES,
    TEMPLATES,
    new_rule,
    clean_rule,
    describe,
    run_scan,
)
from services.market_data import get_universe

router = APIRouter(prefix="/api", tags=["Screener"])


@router.get("/rules/catalogue")
def get_rule_catalogue():
    """Returns all available technical indicator rules and parameter specifications."""
    return {"rules": RULES}


@router.get("/templates")
def get_templates():
    """Returns the ready-made institutional setup templates."""
    out = {}
    for name, t in TEMPLATES.items():
        rules = [new_rule(k, **v) for k, v in t["rules"]]
        out[name] = {
            "name": name,
            "description": t.get("desc", ""),
            "rules": rules,
        }
    return out


@router.get("/setups")
def get_setups():
    """Loads saved setups from Supabase (or in-memory store if db not yet connected)."""
    try:
        rows = load_setups_from_db()
        # If database is empty, seed with the default institutional templates for immediate usability
        if not rows:
            seeded = []
            for name, t in TEMPLATES.items():
                rules = [new_rule(k, **v) for k, v in t["rules"]]
                seeded.append({
                    "id": len(seeded) + 1,
                    "name": name,
                    "price_min": 20.0,
                    "price_max": 50000.0,
                    "visual_rules": t.get("desc", ""),
                    "rule_config": {"version": 2, "rules": rules, "min_pass": 0},
                    "owner": "admin",
                })
            return seeded
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch setups: {str(e)}")


@router.post("/setups")
def save_setup(payload: SetupCreateRequest):
    """Saves or updates a scanner setup in Supabase."""
    try:
        # Validate and clean each rule
        cleaned_rules = [x for x in (clean_rule(r) for r in payload.rules) if x]
        setup_dict = {
            "id": payload.id,
            "name": payload.name,
            "price_min": payload.price_min,
            "price_max": payload.price_max,
            "visual_rules": payload.visual_rules,
            "rule_config": {
                "version": 2,
                "min_pass": payload.min_pass or 0,
                "rules": cleaned_rules,
            },
            "owner": payload.owner or "admin",
        }
        saved = save_setup_to_db(setup_dict)
        return saved
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save setup: {str(e)}")


@router.delete("/setups/{setup_id}")
def delete_setup(setup_id: int):
    """Deletes a setup by ID from Supabase."""
    try:
        success = delete_setup_from_db(setup_id)
        if not success:
            raise HTTPException(status_code=404, detail="Setup not found")
        return {"status": "deleted", "id": setup_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete setup: {str(e)}")


@router.post("/scan", response_model=ScanResponse)
def run_screening_scan(req: ScanRequest):
    """Executes the exact stock screening math and returns matching stocks."""
    try:
        universe_symbols, _ = get_universe(req.universe, req.custom_symbols)
        if not universe_symbols:
            raise HTTPException(status_code=400, detail="Universe symbol list is empty")

        cleaned_rules = [x for x in (clean_rule(r) for r in req.rules) if x]
        if not cleaned_rules:
            raise HTTPException(status_code=400, detail="No valid rules specified for scan")

        results = run_scan(
            universe=universe_symbols,
            rules=cleaned_rules,
            pmin=req.price_min,
            pmax=req.price_max,
            min_turnover=req.min_turnover,
            want_near=req.want_near,
            min_pass=req.min_pass,
        )

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scanning failed: {str(e)}")

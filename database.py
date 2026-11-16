import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("arka.database")
IST = timezone(timedelta(hours=5, minutes=30))

_SUPABASE_CLIENT = None
_LOCAL_SETUPS: List[Dict[str, Any]] = []


def init_supabase():
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "").strip()

    if url and key:
        try:
            from supabase import create_client, Client
            _SUPABASE_CLIENT = create_client(url, key)
            logger.info("Connected to Supabase PostgreSQL database.")
            return _SUPABASE_CLIENT
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            _SUPABASE_CLIENT = None
    else:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set. Using in-memory setup storage.")

    return None


def get_supabase():
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        return init_supabase()
    return _SUPABASE_CLIENT


def load_setups_from_db() -> List[Dict[str, Any]]:
    client = get_supabase()
    if client is not None:
        try:
            res = client.table("setups").select("*").order("created_at", desc=True).execute()
            rows = res.data or []
            # remove large binary image payload from listing for performance
            for r in rows:
                r.pop("reference_image_b64", None)
            return rows
        except Exception as e:
            logger.error(f"Error reading setups from Supabase: {e}")

    # Fallback to local memory
    return [dict(r) for r in _LOCAL_SETUPS]


def save_setup_to_db(setup_data: Dict[str, Any]) -> Dict[str, Any]:
    client = get_supabase()
    sid = setup_data.get("id")

    payload = {
        "name": setup_data.get("name", "Untitled Setup").strip(),
        "price_min": float(setup_data.get("price_min", 0.0)),
        "price_max": float(setup_data.get("price_max", 99999.0)),
        "rsi_min": float(setup_data.get("rsi_min", 0.0)),
        "rsi_max": float(setup_data.get("rsi_max", 100.0)),
        "volume_multiplier": float(setup_data.get("volume_multiplier", 0.0)),
        "atr_pct_min": float(setup_data.get("atr_pct_min", 0.0)),
        "atr_pct_max": float(setup_data.get("atr_pct_max", 0.0)),
        "move_pct_min": float(setup_data.get("move_pct_min", 0.0)),
        "move_pct_max": float(setup_data.get("move_pct_max", 0.0)),
        "visual_rules": setup_data.get("visual_rules", ""),
        "rule_config": setup_data.get("rule_config", {"version": 2, "rules": []}),
        "owner": setup_data.get("owner", "admin"),
    }

    if client is not None:
        try:
            if sid:
                res = client.table("setups").update(payload).eq("id", sid).execute()
            else:
                res = client.table("setups").insert(payload).execute()
            saved = (res.data or [{}])[0]
            return saved
        except Exception as e:
            logger.error(f"Supabase write error: {e}")
            raise e

    # In-memory fallback
    global _LOCAL_SETUPS
    if sid:
        for r in _LOCAL_SETUPS:
            if r["id"] == sid:
                r.update(payload)
                return r
    new_id = max([r.get("id", 0) for r in _LOCAL_SETUPS] + [0]) + 1
    new_row = {
        **payload,
        "id": new_id,
        "created_at": datetime.now(IST).isoformat(),
    }
    _LOCAL_SETUPS.insert(0, new_row)
    return new_row


def delete_setup_from_db(setup_id: int) -> bool:
    client = get_supabase()
    if client is not None:
        try:
            client.table("setups").delete().eq("id", setup_id).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase delete error: {e}")
            raise e

    global _LOCAL_SETUPS
    _LOCAL_SETUPS = [r for r in _LOCAL_SETUPS if r.get("id") != setup_id]
    return True

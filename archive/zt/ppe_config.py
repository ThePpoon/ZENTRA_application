"""
ppe_config.py — ZENTRA PPE Module Selector
==========================================
กำหนดว่าจะตรวจจับ PPE ชิ้นไหนบ้าง
แก้ไขไฟล์นี้แทนที่จะแก้ config.py หลัก

วิธีใช้ใน main.py / modules/ppe.py:
    from ppe_config import PPE_PROFILE

ตัวอย่าง preset ที่มี:
    PPE_PROFILE = "full"          # ตรวจทุกอย่าง (default)
    PPE_PROFILE = "helmet_vest"   # แค่หมวก + เสื้อกั๊ก
    PPE_PROFILE = "helmet_only"   # แค่หมวก
    PPE_PROFILE = "custom"        # ดูที่ CUSTOM_ITEMS ด้านล่าง
==========================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ================================================================
# ── เลือก PROFILE ตรงนี้ ──────────────────────────────────────
# ================================================================

PPE_PROFILE = "custom"#"full" 
# ตัวเลือก:
#   "full"           — ตรวจทุกอย่าง (helmet, vest, gloves, goggles, boots)
#   "helmet_vest"    — แค่หมวก + เสื้อกั๊ก (พื้นฐานสุด)
#   "helmet_only"    — แค่หมวก
#   "helmet_goggles" — หมวก + แว่นตา
#   "vest_boots"     — เสื้อกั๊ก + รองเท้า
#   "no_gloves_boots"— ไม่ตรวจถุงมือและรองเท้า (เน้น visible PPE)
#   "custom"         — กำหนดเองใน CUSTOM_ITEMS

# ================================================================
# ── กำหนดเอง (ใช้เมื่อ PPE_PROFILE = "custom") ───────────────
# ================================================================
CUSTOM_ITEMS = {
    "helmet":       True,
    "vest":         True,
    "gloves":       False,
    "goggles":      True,
    "safety_boots": False,
}


# ================================================================
# ── Alert threshold ───────────────────────────────────────────
# ================================================================
# ถ้าขาด PPE กี่ชิ้นถึงจะ alert (0 = alert ทันทีถ้าขาดชิ้นใดชิ้นหนึ่ง)
MIN_VIOLATIONS_TO_ALERT = 0

# ================================================================
# Profile definitions — ไม่ต้องแก้ส่วนนี้
# ================================================================
_PROFILES: dict[str, dict[str, bool]] = {
    "full": {
        "helmet":        True,
        "vest":          True,
        "gloves":        True,
        "goggles":       True,
        "safety_boots":  True,
    },
    "helmet_vest": {
        "helmet":        True,
        "vest":          True,
        "gloves":        False,
        "goggles":       False,
        "safety_boots":  False,
    },
    "helmet_only": {
        "helmet":        True,
        "vest":          False,
        "gloves":        False,
        "goggles":       False,
        "safety_boots":  False,
    },
    "helmet_goggles": {
        "helmet":        True,
        "vest":          False,
        "gloves":        False,
        "goggles":       True,
        "safety_boots":  False,
    },
    "vest_boots": {
        "helmet":        False,
        "vest":          True,
        "gloves":        False,
        "goggles":       False,
        "safety_boots":  True,
    },
    "no_gloves_boots": {
        "helmet":        True,
        "vest":          True,
        "gloves":        False,
        "goggles":       True,
        "safety_boots":  False,
    },
    "custom": CUSTOM_ITEMS,
}

# ── PPE item metadata ──────────────────────────────────────────
_PPE_META = {
    "helmet":       {"label": "Helmet",    "label_th": "หมวกนิรภัย"},
    "vest":         {"label": "Vest",      "label_th": "เสื้อกั๊กสะท้อนแสง"},
    "gloves":       {"label": "Gloves",    "label_th": "ถุงมือ"},
    "goggles":      {"label": "Goggles",   "label_th": "แว่นตานิรภัย"},
    "safety_boots": {"label": "Boots",     "label_th": "รองเท้าบูท"},
}

# ================================================================
# Public API
# ================================================================

@dataclass
class PPEItem:
    key:        str
    label:      str
    label_th:   str
    enabled:    bool
    # Derived class names
    ok_class:   str = field(init=False)
    ng_class:   str = field(init=False)

    def __post_init__(self):
        self.ok_class = self.key             # e.g. "helmet"
        self.ng_class = f"no_{self.key}"    # e.g. "no_helmet"


def get_active_profile() -> dict[str, bool]:
    """คืน dict {ppe_item: enabled} ของ profile ปัจจุบัน"""
    if PPE_PROFILE not in _PROFILES:
        raise ValueError(f"PPE_PROFILE ไม่รู้จัก: '{PPE_PROFILE}' "
                         f"(เลือกได้: {list(_PROFILES.keys())})")
    return dict(_PROFILES[PPE_PROFILE])


def get_active_items() -> list[PPEItem]:
    """คืน list ของ PPEItem ที่เปิดใช้งาน"""
    profile = get_active_profile()
    items   = []
    for key, meta in _PPE_META.items():
        items.append(PPEItem(
            key       = key,
            label     = meta["label"],
            label_th  = meta["label_th"],
            enabled   = profile.get(key, False),
        ))
    return items


def get_active_classes() -> dict[str, dict]:
    """
    คืน dict แบบเดียวกับ config.PPE_CLASSES แต่กรองตาม profile แล้ว
    ใช้แทนที่ cfg.PPE_CLASSES ใน modules/ppe.py
    """
    profile = get_active_profile()
    classes: dict[str, dict] = {}

    for key, meta in _PPE_META.items():
        enabled = profile.get(key, False)
        if enabled:
            classes[key] = {
                "label":    meta["label"],
                "label_th": meta["label_th"],
                "color":    (0, 210, 0),
                "violation":False,
            }
            ng_key = f"no_{key}"
            classes[ng_key] = {
                "label":    f"No {meta['label']}",
                "label_th": f"ไม่สวม{meta['label_th']}",
                "color":    (0, 0, 220),
                "violation":True,
            }

    # person ให้แสดงเสมอ
    classes["person"] = {
        "label":    "Person",
        "label_th": "บุคคล",
        "color":    (255, 190, 0),
        "violation":False,
    }
    return classes


def get_required_ppe() -> set[str]:
    """คืน set ของ PPE class ที่ต้องสวม (เฉพาะที่ enable)"""
    profile = get_active_profile()
    return {key for key, enabled in profile.items() if enabled}


def is_violation(class_name: str,
                 ppe_classes: Optional[dict] = None) -> bool:
    """ตรวจว่า class_name นั้นเป็น violation หรือเปล่า"""
    if ppe_classes is None:
        ppe_classes = get_active_classes()
    return ppe_classes.get(class_name, {}).get("violation", False)


def describe_profile() -> str:
    """คืน string อธิบาย profile ปัจจุบัน สำหรับแสดงบน OSD"""
    items = get_active_items()
    enabled = [i.label_th for i in items if i.enabled]
    if not enabled:
        return "ไม่ตรวจ PPE"
    return "ตรวจ: " + " | ".join(enabled)


def print_profile_summary():
    """แสดงสรุป profile ปัจจุบัน"""
    print(f"\n{'─'*50}")
    print(f"  PPE Profile : {PPE_PROFILE!r}")
    print(f"  {describe_profile()}")
    print(f"{'─'*50}")
    for item in get_active_items():
        status = "✅ เปิด" if item.enabled else "⬜ ปิด"
        print(f"  {status}  {item.label_th:<20} "
              f"({item.ok_class} / {item.ng_class})")
    print(f"{'─'*50}\n")

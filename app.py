from __future__ import annotations

import io
import os
import re
import zipfile
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Streamlit config (compatibil Streamlit 1.53.1)
# ============================================================
st.set_page_config(page_title="GENERATOR DE VIZUALE FOTBAL", layout="wide")

# ============================================================
# Multi-language support
# ============================================================
RO_TEXT = {
    # UI Elements
    "title": "GENERATOR DE VIZUALE FOTBAL",
    "settings": "SETĂRI",
    "voyo_exclusive": "LOGO VOYO EXCLUSIVE",
    "choose_championship": "ALEGE CAMPIONATUL",
    "tagline": "TAGLINE",
    "choose_background": "ALEGE BACKGROUND-UL",
    "resolutions": "REZOLUȚII",
    "team_a_b": "ECHIPA A / ECHIPA B",
    "team_a": "ECHIPA A",
    "team_b": "ECHIPA B",
    "team_a_logo": "LOGO ECHIPA A",
    "team_b_logo": "LOGO ECHIPA B",
    "team_a_name": "NUME ECHIPA A",
    "team_b_name": "NUME ECHIPA B",
    "scale_pct": "Scalare %",
    "x_offset": "Decalaj X",
    "y_offset": "Decalaj Y",
    "date_hour": "DATĂ & ORĂ",
    "choose_date": "ALEGE DATA",
    "no_date": "FĂRĂ DATĂ",
    "select_date": "SELECTEAZĂ DATA",
    "choose_hour": "ALEGE ORA (HH:MM)",
    "preview": "PREVIZUALIZARE VIZUAL",

    # Buttons & Actions
    "export": "🔧 EXPORTĂ",
    "download": "📥 DESCARCĂ",
    "nav_left": "◀",
    "nav_right": "▶",
    "refresh_dirs": "🔄 REFRESH DIRECTOARE",
    "upload_logo_a": "📤 UPLOAD LOGO ECHIPA A",
    "upload_logo_b": "📤 UPLOAD LOGO ECHIPA B",
    "upload_success": "✅ Logo încărcat cu succes: {}",
    "upload_error": "❌ Eroare la încărcare: {}",
    "refresh_success": "✅ Directoare actualizate",

    # Placeholders & Defaults
    "write_tagline": "SCRIE TAGLINE AICI",
    "hour_placeholder": "HH:MM",
    "none_option": "(niciunul)",

    # Messages & Status
    "no_champs": "Nu există campionate în:",
    "no_backgrounds": "Nu există background-uri în:",
    "no_logos": "Nu există logo-uri în:",
    "export_success": "ZIP: {} imagini!",
    "zip_available": "📦 ZIP disponibil pentru descărcare: **{}** ({:.1f} MB)",

    # VOYO Options
    "yes": "DA",
    "no": "NU",
}

EN_TEXT = {
    # UI Elements
    "title": "FOOTBALL VISUAL GENERATOR",
    "settings": "SETTINGS",
    "voyo_exclusive": "VOYO EXCLUSIVE LOGO",
    "choose_championship": "CHOOSE THE CHAMPIONSHIP",
    "tagline": "TAGLINE",
    "choose_background": "CHOOSE THE BACKGROUND",
    "resolutions": "RESOLUTIONS",
    "team_a_b": "TEAM A / TEAM B",
    "team_a": "TEAM A",
    "team_b": "TEAM B",
    "team_a_logo": "TEAM A LOGO",
    "team_b_logo": "TEAM B LOGO",
    "team_a_name": "TEAM A NAME",
    "team_b_name": "TEAM B NAME",
    "scale_pct": "Scale %",
    "x_offset": "X offset",
    "y_offset": "Y offset",
    "date_hour": "DATE & HOUR",
    "choose_date": "CHOOSE THE DATE",
    "no_date": "NO DATE",
    "select_date": "SELECT DATE",
    "choose_hour": "CHOOSE THE HOUR (HH:MM)",
    "preview": "VISUAL PREVIEW",

    # Buttons & Actions
    "export": "🔧 EXPORT",
    "download": "📥 DOWNLOAD",
    "nav_left": "◀",
    "nav_right": "▶",
    "refresh_dirs": "🔄 REFRESH DIRECTORIES",
    "upload_logo_a": "📤 UPLOAD TEAM A LOGO",
    "upload_logo_b": "📤 UPLOAD TEAM B LOGO",
    "upload_success": "✅ Logo uploaded successfully: {}",
    "upload_error": "❌ Upload error: {}",
    "refresh_success": "✅ Directories refreshed",

    # Placeholders & Defaults
    "write_tagline": "WRITE THE TAGLINE HERE",
    "hour_placeholder": "HH:MM",
    "none_option": "(none)",

    # Messages & Status
    "no_champs": "No championships in:",
    "no_backgrounds": "No backgrounds in:",
    "no_logos": "No logos in:",
    "export_success": "ZIP: {} images!",
    "zip_available": "📦 ZIP available for download: **{}** ({:.1f} MB)",

    # VOYO Options
    "yes": "YES",
    "no": "NO",
}

def get_text(key):
    """Returnează textul în limba curentă"""
    lang = st.session_state.get("language", "RO")
    text_dict = RO_TEXT if lang == "RO" else EN_TEXT
    return text_dict.get(key, key)

# ============================================================
# Language selector component
# ============================================================
def create_language_selector():
    """Creează selectorul de limbă în colțul stânga sus"""
    # Creează un layout cu două coloane
    cols = st.columns([7, 1])
    
    with cols[1]:
        # Folosim un container pentru styling
        with st.container():
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            
            # Limba curentă
            current_lang = st.session_state.get("language", "RO")
            
            # Folosim radio buttons orizontali pentru o selecție mai clară
            col1, col2 = st.columns(2)
            with col1:
                ro_selected = st.button("RO", key="btn_ro",
                                      type="primary" if current_lang == "RO" else "secondary",
                                      use_container_width=True)
            with col2:
                en_selected = st.button("EN", key="btn_en",
                                      type="primary" if current_lang == "EN" else "secondary",
                                      use_container_width=True)
            
            # Verifică care buton a fost apăsat
            if ro_selected and current_lang != "RO":
                st.session_state["language"] = "RO"
                st.rerun()
            elif en_selected and current_lang != "EN":
                st.session_state["language"] = "EN"
                st.rerun()

# ============================================================
# Paths / folders
# ============================================================
APP_DIR = Path(__file__).resolve().parent
CHAMP_DIR = APP_DIR / "CHAMPIONSHIPS"
VOYO_PATH = APP_DIR / "voyoexclusive.png"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# ============================================================
# RO months
# ============================================================
RO_MONTHS = {
    1: "IANUARIE",
    2: "FEBRUARIE",
    3: "MARTIE",
    4: "APRILIE",
    5: "MAI",
    6: "IUNIE",
    7: "IULIE",
    8: "AUGUST",
    9: "SEPTEMBRIE",
    10: "OCTOMBRIE",
    11: "NOIEMBRIE",
    12: "DECEMBRIE",
}

# ============================================================
# Resolution config
# ============================================================
@dataclass(frozen=True)
class ResCfg:
    name: str
    w: int
    h: int
    layout_type: str = "CENTERED"  # CENTERED, SIDE_BY_SIDE, SLIM_BANNER

    voyo_xy: Tuple[int, int] = (0, 0)
    voyo_wh: Tuple[int, int] = (0, 0)

    teamA_xy: Tuple[int, int] = (0, 0)
    teamB_xy: Tuple[int, int] = (0, 0)
    team_logo_w: int = 0

    champ_xy: Tuple[int, int] = (0, 0)
    champ_w: int = 0


RESOLUTIONS: List[ResCfg] = [
    ResCfg(
        "1200 x 1200",
        1200,
        1200,
        layout_type="CENTERED",
        voyo_xy=(375, 110),
        voyo_wh=(450, 50),
        teamA_xy=(100, 350),
        teamB_xy=(700, 350),
        team_logo_w=400,
        champ_xy=(515, 520),
        champ_w=170,
    ),
    ResCfg(
        "1200 x 628",
        1200,
        628,
        layout_type="CENTERED",
        voyo_xy=(436, 50),
        voyo_wh=(328, 36),
        teamA_xy=(150, 180),
        teamB_xy=(722, 180),
        team_logo_w=328,
        champ_xy=(518, 290),
        champ_w=164,
    ),
    ResCfg(
        "1200 x 630",
        1200,
        630,
        layout_type="CENTERED",
        voyo_xy=(436, 50),
        voyo_wh=(328, 36),
        teamA_xy=(150, 180),
        teamB_xy=(722, 180),
        team_logo_w=328,
        champ_xy=(518, 290),
        champ_w=164,
    ),
    ResCfg(
        "1920 x 1080",
        1920,
        1080,
        layout_type="CENTERED",
        voyo_xy=(712, 66),
        voyo_wh=(497, 55),
        teamA_xy=(210, 296),
        teamB_xy=(1216, 296),
        team_logo_w=490,
        champ_xy=(866, 472),
        champ_w=183,
    ),
    ResCfg(
        "1080 x 1080",
        1080,
        1080,
        layout_type="CENTERED",
        voyo_xy=(338, 99),
        voyo_wh=(405, 45),
        teamA_xy=(90, 315),
        teamB_xy=(630, 315),
        team_logo_w=360,
        champ_xy=(463, 468),
        champ_w=153,
    ),
    ResCfg(
        "1080 x 1350",
        1080,
        1350,
        layout_type="CENTERED",
        voyo_xy=(338, 124),
        voyo_wh=(405, 45),
        teamA_xy=(90, 394),
        teamB_xy=(630, 394),
        team_logo_w=360,
        champ_xy=(463, 585),
        champ_w=153,
    ),
    ResCfg(
        "1080 x 1920",
        1080,
        1920,
        layout_type="CENTERED",
        voyo_xy=(270, 70),
        voyo_wh=(540, 60),
        teamA_xy=(148, 400),
        teamB_xy=(674, 400),
        team_logo_w=258,
        champ_xy=(430, 408),
        champ_w=220,
    ),
    ResCfg(
        "970 x 250",
        970,
        250,
        layout_type="SIDE_BY_SIDE",
        voyo_xy=(30, 30),
        voyo_wh=(250, 28),
        teamA_xy=(630, 30),
        teamB_xy=(790, 30),
        team_logo_w=150,
        champ_xy=(740, 85),
        champ_w=70,
    ),
    ResCfg(
        "728 x 90",
        728,
        90,
        layout_type="SLIM_BANNER",
        voyo_xy=(20, 30),
        voyo_wh=(150, 16),
    ),
    ResCfg(
        "970 x 90",
        970,
        90,
        layout_type="SLIM_BANNER",
        voyo_xy=(20, 30),
        voyo_wh=(180, 20),
    ),
]
RES_BY_NAME: Dict[str, ResCfg] = {r.name: r for r in RESOLUTIONS}

# ============================================================
# Utils: list files
# ============================================================
def list_dirs(p: Path) -> List[str]:
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])


def list_images(p: Path) -> List[str]:
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS])


def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\(\)\[\]\.]", "", s, flags=re.UNICODE)
    return s.strip().replace(" ", "_")


# ============================================================
# Fonts / text helpers
# ============================================================
def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates: List[str] = []
    if os.name == "nt":
        candidates += [
            (r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
            r"C:\Windows\Fonts\Arial.ttf",
        ]
    candidates += [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, txt: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if not txt:
        return (0, 0)
    b = draw.textbbox((0, 0), txt, font=font)
    return (b[2] - b[0], b[3] - b[1])


def wrap_text(draw: ImageDraw.ImageDraw, txt: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    txt = " ".join((txt or "").split())
    if not txt:
        return []
    words = txt.split(" ")
    lines: List[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        cw, _ = text_size(draw, cand, font)
        if cw <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_tagline(draw: ImageDraw.ImageDraw, tagline: str, base_font_px: int, max_w: int) -> Tuple[ImageFont.ImageFont, List[str]]:
    tagline = (tagline or "").strip()
    if not tagline:
        return load_font(base_font_px, bold=True), []

    size = max(10, base_font_px)
    while size >= 10:
        font = load_font(size, bold=True)
        lines = wrap_text(draw, tagline, font, max_w)
        if len(lines) <= 2 and all(text_size(draw, ln, font)[0] <= max_w for ln in lines):
            return font, lines
        size -= 1

    font = load_font(10, bold=True)
    lines = wrap_text(draw, tagline, font, max_w)[:2]
    return font, lines


def fmt_ro_date(d: date) -> str:
    return f"{d.day} {RO_MONTHS.get(d.month, str(d.month)).upper()}"


def valid_hour(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d{1,2})\s*:\s*(\d{2})", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return f"{hh:02d}:{mm:02d}"


# ============================================================
# Images
# ============================================================
@st.cache_data(show_spinner=False)
def load_rgba(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def resize_keep_aspect(img: Image.Image, target_w: int) -> Image.Image:
    w, h = img.size
    if w <= 0:
        return img
    scale = target_w / w
    nh = max(1, int(round(h * scale)))
    return img.resize((target_w, nh), Image.LANCZOS)


def paste_rgba(dst: Image.Image, src: Image.Image, xy: Tuple[int, int]) -> None:
    dst.alpha_composite(src, dest=xy)


# ============================================================
# File upload functions
# ============================================================
def save_uploaded_file(uploaded_file, destination_folder: Path) -> Optional[Path]:
    """Salvează un fișier încărcat în folderul specificat"""
    if uploaded_file is None:
        return None

    try:
        # Asigură-te că folderul există
        destination_folder.mkdir(parents=True, exist_ok=True)
        
        # Creează numele fișierului (curățat)
        original_name = Path(uploaded_file.name)
        safe_filename = safe_name(original_name.stem) + original_name.suffix.lower()
        file_path = destination_folder / safe_filename
        
        # Salvează fișierul
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(get_text("upload_error").format(str(e)))
        return None


def refresh_directories():
    """Forțează reîncărcarea listelor de fișiere"""
    # Curăță cache-ul pentru funcțiile de încărcare imagini
    st.cache_data.clear()
    # Afișează mesaj de succes
    st.success(get_text("refresh_success"))
    # Reîncarcă pagina pentru a actualiza listele
    st.rerun()


# ============================================================
# Per-resolution state
# ============================================================
def default_res_state() -> Dict:
    return {
        "teamA_logo": None,
        "teamB_logo": None,
        "teamA_scale": 100,
        "teamB_scale": 100,
        "teamA_dx": 0,
        "teamA_dy": 0,
        "teamB_dx": 0,
        "teamB_dy": 0,
    }


def ensure_res_state(res_name: str) -> None:
    if "per_res" not in st.session_state:
        st.session_state["per_res"] = {}
    if res_name not in st.session_state["per_res"]:
        st.session_state["per_res"][res_name] = default_res_state()
        # Propagă globalele către noua rezoluție
        gA = st.session_state.get("global_teamA_logo")
        gB = st.session_state.get("global_teamB_logo")
        if gA:
            st.session_state["per_res"][res_name]["teamA_logo"] = gA
        if gB:
            st.session_state["per_res"][res_name]["teamB_logo"] = gB


def update_global_logos():
    """Actualizează logo-urile globale pentru toate rezoluțiile selectate"""
    selected_res = get_selected_res()
    gA = st.session_state.get("global_teamA_logo")
    gB = st.session_state.get("global_teamB_logo")

    for rn in selected_res:
        ensure_res_state(rn)
        per = st.session_state["per_res"][rn]
        per["teamA_logo"] = gA
        per["teamB_logo"] = gB


def pick_default_bg(bgs: List[str]) -> Optional[str]:
    if not bgs:
        return None
    for x in bgs:
        if x.lower().startswith("bg1"):
            return x
    return bgs[0]


# ============================================================
# Render visual
# ============================================================
def _calc_logo_height_for_date_anchor(
    logo_path: Optional[Path],
    base_width_px: int,
    scale_factor: float,
) -> int:
    """Returnează înălțimea logo-ului după resize, pentru calculul ancorei DATE/ORĂ."""
    if not logo_path or not logo_path.exists():
        return 0
    img = load_rgba(str(logo_path))
    w = max(1, int(round(base_width_px * scale_factor)))
    img = resize_keep_aspect(img, w)
    return int(img.size[1])


def render_visual(
    cfg: ResCfg,
    championship_path: Path,
    background_path: Optional[Path],
    voyo_on: bool,
    tagline: str,
    teamA_logo_path: Optional[Path],
    teamB_logo_path: Optional[Path],
    teamA_scale: int,
    teamB_scale: int,
    teamA_dx: int,
    teamA_dy: int,
    teamB_dx: int,
    teamB_dy: int,
    chosen_date: Optional[date],
    hour_hhmm: Optional[str],
    teamA_name: str = "",
    teamB_name: str = "",
) -> Image.Image:
    if background_path and background_path.exists():
        bg = load_rgba(str(background_path)).resize((cfg.w, cfg.h), Image.LANCZOS)
        canvas = bg.copy()
    else:
        canvas = Image.new("RGBA", (cfg.w, cfg.h), (0, 0, 0, 0))

    draw = ImageDraw.Draw(canvas)

    if cfg.layout_type == "SLIM_BANNER":
        # Layout specific pentru bannere subțiri (728x90, 970x90)
        # Voyo logo pe stânga
        if voyo_on and VOYO_PATH.exists():
            voyo = load_rgba(str(VOYO_PATH)).resize(cfg.voyo_wh, Image.LANCZOS)
            paste_rgba(canvas, voyo, cfg.voyo_xy)

        # Text echipe și dată în restul spațiului
        font_size = int(cfg.h * 0.4)
        font = load_font(font_size, bold=True)

        match_text = f"{teamA_name} - {teamB_name}".upper()
        if chosen_date:
            date_str = fmt_ro_date(chosen_date)
            if hour_hhmm:
                date_str += f" | ORA {hour_hhmm}"
            match_text += f" | {date_str}"

        tw, th = text_size(draw, match_text, font)
        # Centrat pe restul lățimii după logo voyo
        start_x = cfg.voyo_xy[0] + cfg.voyo_wh[0] + 40
        draw.text((start_x, (cfg.h - th) / 2), match_text, font=font, fill=(255, 255, 255, 255))

        return canvas

    if cfg.layout_type == "SIDE_BY_SIDE":
        # Layout specific pentru 970x250
        # Voyo și Tagline pe stânga
        if voyo_on and VOYO_PATH.exists():
            voyo = load_rgba(str(VOYO_PATH)).resize(cfg.voyo_wh, Image.LANCZOS)
            paste_rgba(canvas, voyo, cfg.voyo_xy)

        tagline = (tagline or "").strip()
        if tagline:
            font = load_font(int(cfg.voyo_wh[1] * 1.5), bold=True)
            draw.text((cfg.voyo_xy[0], cfg.voyo_xy[1] + cfg.voyo_wh[1] + 20), tagline.upper(), font=font, fill=(255, 255, 255, 255))

        # Data și Ora sub tagline
        if chosen_date:
            date_txt = fmt_ro_date(chosen_date)
            if hour_hhmm:
                date_txt += f" | ORA {hour_hhmm}"
            font = load_font(int(cfg.voyo_wh[1] * 1.2), bold=True)
            draw.text((cfg.voyo_xy[0], cfg.h - 60), date_txt, font=font, fill=(255, 255, 255, 255))

        # Logo-uri echipe grupate în dreapta
        if teamA_logo_path and teamA_logo_path.exists():
            img = load_rgba(str(teamA_logo_path))
            w = max(1, int(round(cfg.team_logo_w * (teamA_scale / 100.0))))
            img = resize_keep_aspect(img, w)
            paste_rgba(canvas, img, (cfg.teamA_xy[0] + teamA_dx, cfg.teamA_xy[1] + teamA_dy))

        if teamB_logo_path and teamB_logo_path.exists():
            img = load_rgba(str(teamB_logo_path))
            w = max(1, int(round(cfg.team_logo_w * (teamB_scale / 100.0))))
            img = resize_keep_aspect(img, w)
            paste_rgba(canvas, img, (cfg.teamB_xy[0] + teamB_dx, cfg.teamB_xy[1] + teamB_dy))

        # Logo campionat între ele
        champ_logo = championship_path / "logocampionat.png"
        if champ_logo.exists():
            img = resize_keep_aspect(load_rgba(str(champ_logo)), cfg.champ_w)
            paste_rgba(canvas, img, cfg.champ_xy)

        return canvas

    # DEFAULT CENTERED LAYOUT
    # VOYO
    voyo_h = cfg.voyo_wh[1]
    if voyo_on and VOYO_PATH.exists():
        voyo = load_rgba(str(VOYO_PATH)).resize(cfg.voyo_wh, Image.LANCZOS)
        paste_rgba(canvas, voyo, cfg.voyo_xy)

    # TAGLINE (max 2 linii, centrat sub VOYO)
    tagline = (tagline or "").strip()
    if tagline:
        base_font_px = int(round(voyo_h * 1.20))
        max_w = cfg.w - 80
        font, lines = fit_tagline(draw, tagline, base_font_px, max_w)
        if lines:
            cx = cfg.voyo_xy[0] + cfg.voyo_wh[0] / 2
            y = cfg.voyo_xy[1] + cfg.voyo_wh[1] + 20
            line_h = int(round(text_size(draw, "Ag", font)[1] * 1.10))
            for i, ln in enumerate(lines[:2]):
                tw, _ = text_size(draw, ln, font)
                draw.text((cx - tw / 2, y + i * line_h), ln, font=font, fill=(255, 255, 255, 255))

    # Logo campionat
    champ_logo = championship_path / "logocampionat.png"
    if champ_logo.exists():
        img = resize_keep_aspect(load_rgba(str(champ_logo)), cfg.champ_w)
        paste_rgba(canvas, img, cfg.champ_xy)

    # Logo-uri echipe
    # Echipa A
    if teamA_logo_path and teamA_logo_path.exists():
        img = load_rgba(str(teamA_logo_path))
        w = max(1, int(round(cfg.team_logo_w * (teamA_scale / 100.0))))
        img = resize_keep_aspect(img, w)
        x = cfg.teamA_xy[0] + teamA_dx
        y = cfg.teamA_xy[1] + teamA_dy
        paste_rgba(canvas, img, (x, y))

    # Echipa B (85% scale + y+30)
    if teamB_logo_path and teamB_logo_path.exists():
        img = load_rgba(str(teamB_logo_path))
        effective_scale = (teamB_scale / 100.0) * 0.85
        w = max(1, int(round(cfg.team_logo_w * effective_scale)))
        img = resize_keep_aspect(img, w)
        x = cfg.teamB_xy[0] + teamB_dx
        y = cfg.teamB_xy[1] + teamB_dy + 30
        paste_rgba(canvas, img, (x, y))

    # ============================================================
    # Dată + Oră
    # ============================================================
    if chosen_date:
        date_txt = fmt_ro_date(chosen_date)
        date_font_px = max(10, cfg.voyo_wh[1])
        date_font = load_font(date_font_px, bold=True)

        hA = _calc_logo_height_for_date_anchor(
            teamA_logo_path,
            base_width_px=cfg.team_logo_w,
            scale_factor=(teamA_scale / 100.0),
        )
        hB = _calc_logo_height_for_date_anchor(
            teamB_logo_path,
            base_width_px=cfg.team_logo_w,
            scale_factor=(teamB_scale / 100.0) * 0.85,
        )

        bottoms: List[int] = []
        if hA > 0:
            bottoms.append(cfg.teamA_xy[1] + hA)
        if hB > 0:
            bottoms.append(cfg.teamB_xy[1] + 30 + hB)

        if bottoms:
            y_base = max(bottoms) + 30
        else:
            y_base = int(cfg.h * 0.60)

        cx = cfg.w / 2
        tw, th = text_size(draw, date_txt, date_font)
        draw.text((cx - tw / 2, y_base), date_txt, font=date_font, fill=(255, 255, 255, 255))

        if hour_hhmm:
            hour_txt = f"ORA {hour_hhmm}"
            hour_font = load_font(max(10, int(round(date_font_px * 0.85))), bold=True)
            hw, _ = text_size(draw, hour_txt, hour_font)
            draw.text((cx - hw / 2, y_base + th + 10), hour_txt, font=hour_font, fill=(255, 255, 255, 255))

    return canvas


# ============================================================
# Helpers: selections + propagation
# ============================================================
def get_selected_res() -> List[str]:
    sel_map = st.session_state.get("res_selected", {})
    selected = [name for name, v in sel_map.items() if v]
    return selected or [RESOLUTIONS[0].name]


# ============================================================
# Callback functions
# ============================================================
def clean_team_name(filename: Optional[str]) -> str:
    if not filename:
        return ""
    name = Path(filename).stem
    # Înlocuiește underscore cu spațiu și pune litere mari
    name = name.replace("_", " ").upper()
    return name


def team_a_logo_callback():
    """Callback pentru schimbarea logo-ului Echipa A"""
    new_val = st.session_state.teamA_selectbox
    newA = None if new_val == get_text("none_option") else new_val
    
    # Dacă logo-ul Echipa B este același, resetează-l
    if newA and newA == st.session_state.get("global_teamB_logo"):
        st.session_state["global_teamB_logo"] = None
    
    st.session_state["global_teamA_logo"] = newA
    # Auto-fill nume echipă
    if newA:
        st.session_state["teamA_name_val"] = clean_team_name(newA)
    
    update_global_logos()


def team_b_logo_callback():
    """Callback pentru schimbarea logo-ului Echipa B"""
    new_val = st.session_state.teamB_selectbox
    newB = None if new_val == get_text("none_option") else new_val
    
    # Dacă logo-ul Echipa A este același, resetează-l
    if newB and newB == st.session_state.get("global_teamA_logo"):
        st.session_state["global_teamA_logo"] = None
    
    st.session_state["global_teamB_logo"] = newB
    # Auto-fill nume echipă
    if newB:
        st.session_state["teamB_name_val"] = clean_team_name(newB)
        
    update_global_logos()


def date_mode_callback():
    """Callback pentru schimbarea modului de dată"""
    new_mode = st.session_state.date_mode_select
    st.session_state["date_mode"] = new_mode
    
    if new_mode == get_text("no_date"):
        st.session_state["chosen_date"] = None


# ============================================================
# UI
# ============================================================
# Adaugă selectorul de limbă în partea de sus
create_language_selector()

# Titlu principal
st.title(get_text("title"))

# Inițializare session state
if "global_teamA_logo" not in st.session_state:
    st.session_state["global_teamA_logo"] = None
if "global_teamB_logo" not in st.session_state:
    st.session_state["global_teamB_logo"] = None
if "teamA_name_val" not in st.session_state:
    st.session_state["teamA_name_val"] = ""
if "teamB_name_val" not in st.session_state:
    st.session_state["teamB_name_val"] = ""
if "date_mode" not in st.session_state:
    st.session_state["date_mode"] = get_text("no_date")
if "chosen_date" not in st.session_state:
    st.session_state["chosen_date"] = None
if "hour_raw" not in st.session_state:
    st.session_state["hour_raw"] = ""
if "export_zip_data" not in st.session_state:
    st.session_state["export_zip_data"] = None
if "language" not in st.session_state:
    st.session_state["language"] = "RO"

# -----------------------
# SIDEBAR = LEFT PANEL (scrollable)
# -----------------------
with st.sidebar:
    st.header(get_text("settings"))

    st.subheader(get_text("voyo_exclusive"))
    voyo_choice = st.radio(
        get_text("voyo_exclusive"),
        options=[get_text("yes"), get_text("no")],
        index=0 if st.session_state.get("voyo_choice", get_text("no")) == get_text("yes") else 1,
        horizontal=True,
        key="voyo_choice_radio",
        label_visibility="collapsed",
    )
    st.session_state["voyo_choice"] = voyo_choice

    st.subheader(get_text("choose_championship"))
    champs = sorted(list_dirs(CHAMP_DIR))
    if not champs:
        st.error(f"{get_text('no_champs')} {CHAMP_DIR}")
        st.stop()

    championship_name = st.selectbox(
        get_text("choose_championship"),
        champs,
        key="championship_name_select",
        label_visibility="collapsed",
    )
    if championship_name != st.session_state.get("championship_name"):
        st.session_state["championship_name"] = championship_name
        # Resetare selecții logo când se schimbă campionatul
        st.session_state["global_teamA_logo"] = None
        st.session_state["global_teamB_logo"] = None

    championship_path = CHAMP_DIR / championship_name

    st.subheader(get_text("tagline"))
    tagline_val = st.text_input(
        get_text("tagline"),
        value=st.session_state.get("tagline", ""),
        placeholder=get_text("write_tagline"),
        key="tagline_input",
        label_visibility="collapsed",
    )
    st.session_state["tagline"] = tagline_val

    st.subheader(get_text("choose_background"))
    bg_dir = championship_path / "BACKGROUNDS"
    bgs = list_images(bg_dir)
    if not bgs:
        st.warning(f"{get_text('no_backgrounds')} {bg_dir}")
        bg_choice = None
    else:
        if "bg_choice" not in st.session_state or st.session_state.get("bg_choice") not in bgs:
            st.session_state["bg_choice"] = pick_default_bg(bgs)
        bg_choice = st.selectbox(get_text("choose_background"), bgs, key="bg_choice_select", label_visibility="collapsed")
    st.session_state["bg_choice"] = bg_choice

    st.subheader(get_text("resolutions"))
    if "res_selected" not in st.session_state:
        st.session_state["res_selected"] = {r.name: True for r in RESOLUTIONS}

    for r in RESOLUTIONS:
        st.session_state["res_selected"][r.name] = st.checkbox(
            r.name,
            value=st.session_state["res_selected"].get(r.name, True),
            key=f"chk_{r.name}",
        )

    selected_res = get_selected_res()

    st.session_state.setdefault("preview_idx", 0)
    if st.session_state["preview_idx"] >= len(selected_res):
        st.session_state["preview_idx"] = 0

    current_res_name = selected_res[st.session_state["preview_idx"]]
    ensure_res_state(current_res_name)
    cur = st.session_state["per_res"][current_res_name]

    st.divider()
    st.subheader(get_text("team_a_b"))

    logos_dir = championship_path / "LOGOS TEAM"
    team_logos = list_images(logos_dir)
    if not team_logos:
        st.warning(f"{get_text('no_logos')} {logos_dir}")
    else:
        # Obține selecțiile globale curente
        gA = st.session_state.get("global_teamA_logo")
        gB = st.session_state.get("global_teamB_logo")
        
        # Creează opțiuni excluzând selecția echipei opuse
        none_option = get_text("none_option")
        a_options = [none_option] + [x for x in team_logos if x != gB]
        b_options = [none_option] + [x for x in team_logos if x != gA]
        
        st.markdown(f"#### {get_text('team_a')}")
        
        # Găsește indexul curent pentru Echipa A
        a_index = 0
        if gA and gA in a_options:
            a_index = a_options.index(gA)
        elif gA and gA not in a_options:
            # Reset dacă selecția este invalidă
            st.session_state["global_teamA_logo"] = None
            gA = None
        
        # Selecție Echipa A - CU CALLBACK
        new_teamA = st.selectbox(
            get_text("team_a_logo"),
            a_options,
            index=a_index,
            key="teamA_selectbox",
            label_visibility="collapsed",
            on_change=team_a_logo_callback
        )

        # Dacă s-a schimbat prin callback, actualizează
        newA = None if new_teamA == none_option else new_teamA
        if newA != gA:
            st.session_state["global_teamA_logo"] = newA
            if newA and newA == st.session_state.get("global_teamB_logo"):
                st.session_state["global_teamB_logo"] = None
            update_global_logos()

        # Nume Echipa A
        teamA_name_val = st.text_input(
            get_text("team_a_name"),
            value=st.session_state.get("teamA_name_val", ""),
            key="teamA_name_input",
        )
        st.session_state["teamA_name_val"] = teamA_name_val
        
        # Controale Echipa A - valori direct din session_state
        col1, col2 = st.columns(2)
        with col1:
            teamA_scale = st.number_input(
                f"{get_text('scale_pct')} ({get_text('team_a')})",
                min_value=10,
                max_value=300,
                value=int(cur.get("teamA_scale", 100)),
                step=1,
                key=f"teamA_scale_{current_res_name}",
            )
            cur["teamA_scale"] = teamA_scale
            
        with col2:
            teamA_dx = st.number_input(
                get_text("x_offset"),
                value=int(cur.get("teamA_dx", 0)),
                step=1,
                key=f"teamA_dx_{current_res_name}",
            )
            cur["teamA_dx"] = teamA_dx
            
        teamA_dy = st.number_input(
            f"{get_text('y_offset')} ({get_text('team_a')})",
            value=int(cur.get("teamA_dy", 0)),
            step=1,
            key=f"teamA_dy_{current_res_name}",
        )
        cur["teamA_dy"] = teamA_dy

        st.markdown(f"#### {get_text('team_b')}")
        
        # Găsește indexul curent pentru Echipa B
        b_index = 0
        if gB and gB in b_options:
            b_index = b_options.index(gB)
        elif gB and gB not in b_options:
            # Reset dacă selecția este invalidă
            st.session_state["global_teamB_logo"] = None
            gB = None
        
        # Selecție Echipa B - CU CALLBACK
        new_teamB = st.selectbox(
            get_text("team_b_logo"),
            b_options,
            index=b_index,
            key="teamB_selectbox",
            label_visibility="collapsed",
            on_change=team_b_logo_callback
        )
        
        # Dacă s-a schimbat prin callback, actualizează
        newB = None if new_teamB == none_option else new_teamB
        if newB != gB:
            st.session_state["global_teamB_logo"] = newB
            if newB and newB == st.session_state.get("global_teamA_logo"):
                st.session_state["global_teamA_logo"] = None
            update_global_logos()

        # Nume Echipa B
        teamB_name_val = st.text_input(
            get_text("team_b_name"),
            value=st.session_state.get("teamB_name_val", ""),
            key="teamB_name_input",
        )
        st.session_state["teamB_name_val"] = teamB_name_val
        
        # Controale Echipa B - valori direct din session_state
        col1, col2 = st.columns(2)
        with col1:
            teamB_scale = st.number_input(
                f"{get_text('scale_pct')} ({get_text('team_b')})",
                min_value=10,
                max_value=300,
                value=int(cur.get("teamB_scale", 100)),
                step=1,
                key=f"teamB_scale_{current_res_name}",
            )
            cur["teamB_scale"] = teamB_scale

        with col2:
            teamB_dx = st.number_input(
                get_text("x_offset"),
                value=int(cur.get("teamB_dx", 0)),
                step=1,
                key=f"teamB_dx_{current_res_name}",
            )
            cur["teamB_dx"] = teamB_dx

        teamB_dy = st.number_input(
            f"{get_text('y_offset')} ({get_text('team_b')})",
            value=int(cur.get("teamB_dy", 0)),
            step=1,
            key=f"teamB_dy_{current_res_name}",
        )
        cur["teamB_dy"] = teamB_dy

    st.divider()
    
    # ============================================================
    # NOI BUTOANE: Refresh și Upload Logo-uri
    # ============================================================
    st.subheader("📁 MANAGEMENT FIȘIERE")
    
    # Buton Refresh Directoare
    if st.button(get_text("refresh_dirs"), key="btn_refresh_dirs", use_container_width=True):
        refresh_directories()
    
    st.markdown("---")
    
    # Upload Logo Echipa A
    st.markdown(f"**{get_text('upload_logo_a')}**")
    uploaded_file_a = st.file_uploader(
        f"{get_text('upload_logo_a')} (PNG/JPG)",
        type=['png', 'jpg', 'jpeg', 'webp'],
        key="upload_logo_a",
        label_visibility="collapsed"
    )
    
    if uploaded_file_a is not None:
        # Verifică dacă există un campionat selectat
        if championship_name:
            logos_dir = championship_path / "LOGOS TEAM"
            saved_path = save_uploaded_file(uploaded_file_a, logos_dir)
            if saved_path:
                st.success(get_text("upload_success").format(saved_path.name))
                # Reîmprospătează listele
                refresh_directories()
        else:
            st.warning("Selectează un campionat înainte de upload")
    
    # Upload Logo Echipa B
    st.markdown(f"**{get_text('upload_logo_b')}**")
    uploaded_file_b = st.file_uploader(
        f"{get_text('upload_logo_b')} (PNG/JPG)",
        type=['png', 'jpg', 'jpeg', 'webp'],
        key="upload_logo_b",
        label_visibility="collapsed"
    )
    
    if uploaded_file_b is not None:
        # Verifică dacă există un campionat selectat
        if championship_name:
            logos_dir = championship_path / "LOGOS TEAM"
            saved_path = save_uploaded_file(uploaded_file_b, logos_dir)
            if saved_path:
                st.success(get_text("upload_success").format(saved_path.name))
                # Reîmprospătează listele
                refresh_directories()
        else:
            st.warning("Selectează un campionat înainte de upload")
    
    st.divider()
    # ============================================================
    # Sfârșit butoane noi
    # ============================================================

    st.subheader(get_text("date_hour"))

    # Selecție DATA - CU CALLBACK
    date_mode = st.selectbox(
        get_text("choose_date"),
        [get_text("no_date"), get_text("select_date")],
        index=0 if st.session_state.get("date_mode", get_text("no_date")) == get_text("no_date") else 1,
        key="date_mode_select",
        on_change=date_mode_callback
    )

    # Asigură sincronizarea dacă callback-ul nu a fost apelat
    if date_mode != st.session_state.get("date_mode"):
        st.session_state["date_mode"] = date_mode
        if date_mode == get_text("no_date"):
            st.session_state["chosen_date"] = None

    if st.session_state["date_mode"] == get_text("select_date"):
        chosen_date_val = st.date_input(
            "Data",
            value=st.session_state.get("chosen_date") or date.today(),
            key="chosen_date_input",
        )
        st.session_state["chosen_date"] = chosen_date_val
    else:
        st.session_state["chosen_date"] = None

    # Input ORĂ - se actualizează automat
    hour_val = st.text_input(
        get_text("choose_hour"),
        value=st.session_state.get("hour_raw", ""),
        placeholder=get_text("hour_placeholder"),
        key="hour_raw_input",
        label_visibility="collapsed",
    )
    st.session_state["hour_raw"] = hour_val

    st.divider()

    # Secțiune export
    col1, col2 = st.columns(2)

    with col1:
        if st.button(get_text("export"), key="btn_export_generate", use_container_width=True):
            # Generează imagini și creează ZIP în memorie
            teamA_name_safe = safe_name(st.session_state.get("teamA_name_val") or st.session_state.get("global_teamA_logo") or "TeamA")
            teamB_name_safe = safe_name(st.session_state.get("teamB_name_val") or st.session_state.get("global_teamB_logo") or "TeamB")
            folder_name = f"{safe_name(championship_name)}_{teamA_name_safe}_vs_{teamB_name_safe}"

            bg_choice = st.session_state.get("bg_choice")
            background_path = (bg_dir / bg_choice) if bg_choice else None

            hour_ok = valid_hour(st.session_state.get("hour_raw", ""))
            chosen_date = st.session_state.get("chosen_date")
            voyo_on = (st.session_state.get("voyo_choice", get_text("no")) == get_text("yes"))
            tagline = st.session_state.get("tagline", "")

            # Creează ZIP în memorie
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for rn in selected_res:
                    ensure_res_state(rn)
                    cfg = RES_BY_NAME[rn]
                    per = st.session_state["per_res"][rn]

                    imgA = (logos_dir / per["teamA_logo"]) if per.get("teamA_logo") else None
                    imgB = (logos_dir / per["teamB_logo"]) if per.get("teamB_logo") else None

                    canvas = render_visual(
                        cfg=cfg,
                        championship_path=championship_path,
                        background_path=background_path,
                        voyo_on=voyo_on,
                        tagline=tagline,
                        teamA_logo_path=imgA,
                        teamB_logo_path=imgB,
                        teamA_scale=int(per.get("teamA_scale", 100)),
                        teamB_scale=int(per.get("teamB_scale", 100)),
                        teamA_dx=int(per.get("teamA_dx", 0)),
                        teamA_dy=int(per.get("teamA_dy", 0)),
                        teamB_dx=int(per.get("teamB_dx", 0)),
                        teamB_dy=int(per.get("teamB_dy", 0)),
                        chosen_date=chosen_date if isinstance(chosen_date, date) else None,
                        hour_hhmm=hour_ok,
                        teamA_name=st.session_state.get("teamA_name_val", ""),
                        teamB_name=st.session_state.get("teamB_name_val", ""),
                    )

                    # Salvează imaginea în BytesIO
                    img_buffer = io.BytesIO()
                    canvas.convert("RGB").save(img_buffer, format="JPEG", quality=95)
                    img_buffer.seek(0)

                    # Adaugă în ZIP
                    file_name = f"{folder_name}/{safe_name(rn)}.jpg"
                    zip_file.writestr(file_name, img_buffer.getvalue())

            # Stochează datele ZIP în session state pentru descărcare
            zip_buffer.seek(0)
            st.session_state["export_zip_data"] = zip_buffer.getvalue()
            st.session_state["export_zip_name"] = f"{folder_name}.zip"
            st.success(get_text("export_success").format(len(selected_res)))

    with col2:
        # Buton descărcare (apare după generarea ZIP)
        if st.session_state.get("export_zip_data"):
            st.download_button(
                label=get_text("download"),
                data=st.session_state["export_zip_data"],
                file_name=st.session_state.get("export_zip_name", "vizuale_fotbal.zip"),
                mime="application/zip",
                key="btn_download_zip",
                use_container_width=True
            )

# -----------------------
# PAGINA PRINCIPALĂ = PREVIEW
# -----------------------
st.subheader(get_text("preview"))

selected_res = get_selected_res()
st.session_state.setdefault("preview_idx", 0)
if st.session_state["preview_idx"] >= len(selected_res):
    st.session_state["preview_idx"] = 0

nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1], gap="small")
with nav_col1:
    if st.button(get_text("nav_left"), key="nav_left", use_container_width=True):
        st.session_state["preview_idx"] = (st.session_state["preview_idx"] - 1) % len(selected_res)

with nav_col2:
    label = f"{selected_res[st.session_state['preview_idx']]} | {st.session_state['preview_idx'] + 1} / {len(selected_res)}"
    st.markdown(
        f"<div style='text-align:center;font-weight:700;font-size:16px;padding-top:8px;'>{label}</div>",
        unsafe_allow_html=True,
    )

with nav_col3:
    if st.button(get_text("nav_right"), key="nav_right", use_container_width=True):
        st.session_state["preview_idx"] = (st.session_state["preview_idx"] + 1) % len(selected_res)

current_res_name = selected_res[st.session_state["preview_idx"]]
current_cfg = RES_BY_NAME[current_res_name]
ensure_res_state(current_res_name)
cur = st.session_state["per_res"][current_res_name]

championship_name = st.session_state.get("championship_name")
if not championship_name:
    st.info("Selectează un campionat în bara laterală.")
    st.stop()

championship_path = CHAMP_DIR / championship_name
logos_dir = championship_path / "LOGOS TEAM"
bg_dir = championship_path / "BACKGROUNDS"

bg_choice = st.session_state.get("bg_choice")
background_path = (bg_dir / bg_choice) if bg_choice else None

voyo_on = (st.session_state.get("voyo_choice", get_text("no")) == get_text("yes"))
tagline = st.session_state.get("tagline", "")

# Asigură-te că logo-urile sunt sincronizate
cur["teamA_logo"] = st.session_state.get("global_teamA_logo")
cur["teamB_logo"] = st.session_state.get("global_teamB_logo")

imgA = (logos_dir / cur["teamA_logo"]) if cur.get("teamA_logo") else None
imgB = (logos_dir / cur["teamB_logo"]) if cur.get("teamB_logo") else None

chosen_date = st.session_state.get("chosen_date")
hour_ok = valid_hour(st.session_state.get("hour_raw", ""))

canvas = render_visual(
    cfg=current_cfg,
    championship_path=championship_path,
    background_path=background_path,
    voyo_on=voyo_on,
    tagline=tagline,
    teamA_logo_path=imgA,
    teamB_logo_path=imgB,
    teamA_scale=int(cur.get("teamA_scale", 100)),
    teamB_scale=int(cur.get("teamB_scale", 100)),
    teamA_dx=int(cur.get("teamA_dx", 0)),
    teamA_dy=int(cur.get("teamA_dy", 0)),
    teamB_dx=int(cur.get("teamB_dx", 0)),
    teamB_dy=int(cur.get("teamB_dy", 0)),
    chosen_date=chosen_date if isinstance(chosen_date, date) else None,
    hour_hhmm=hour_ok,
    teamA_name=st.session_state.get("teamA_name_val", ""),
    teamB_name=st.session_state.get("teamB_name_val", ""),
)

buf = io.BytesIO()
canvas.convert("RGB").save(buf, format="JPEG", quality=92)
st.image(buf.getvalue(), use_container_width=True)

# Informații despre export
if st.session_state.get("export_zip_data"):
    zip_size = len(st.session_state["export_zip_data"]) / (1024 * 1024)  # MB
    st.info(get_text("zip_available").format(
        st.session_state.get('export_zip_name'),
        zip_size
    ))
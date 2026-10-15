"""
dashboard/components/theme.py
Design tokens, railway signage typography, and global CSS injection for YatraGov AI.
Theme: "Departure Board & Ticket" (Indian Railway Signage & Warm Paper).
"""

import streamlit as st

# Color Palette (Departure Board & Ticket)
NAVY = "#0B1F3A"              # Deep Railway Navy (Sidebar, Headings, Board Panels)
NAVY_SURFACE = "#132B4F"      # Elevated Navy for Dark Cards & Active Nav Rows
NAVY_BORDER = "#1E3A63"       # Subtle Navy Border for Panels
SAFFRON = "#FF9F1C"           # Saffron Accent (Active Markers, Buttons, Underlines)
SAFFRON_HOVER = "#F08C00"     # Darker Saffron for Hover
WARM_PAPER = "#FAF6EE"        # Warm Railway Paper Background
WHITE_CARD = "#FFFFFF"        # Pure White Ticket Cards
MUTED_LIGHT = "#5B6B82"       # Muted Slate for Text on Light Backgrounds
MUTED_DARK = "#8FA3BF"        # Light Grey-Blue for Muted Text on Dark Navy Backgrounds
NAVY_TEXT_LIGHT = "#E2E8F0"   # Crisp Silver-White for Nav Links on Dark Navy

# Signal Status Colors (Status only, never decoration)
SIGNAL_GREEN = "#22C55E"      # Normal / Healthy / Confirmed
SIGNAL_AMBER = "#FACC15"      # Warning / Moderate / Caution
SIGNAL_RED = "#EF4444"        # Critical / High Delay / Low Confirmation

# Typography Stacks with System Fallbacks
FONT_HEADINGS = "'Barlow Condensed', 'Arial Narrow', 'Trebuchet MS', 'Impact', sans-serif"
FONT_BODY = "'Barlow', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
FONT_MONO = "'Share Tech Mono', 'Courier New', Courier, monospace"


def apply_theme() -> None:
    """Inject global CSS rules, railway signage fonts, and responsive departure board styles."""
    st.markdown(
        f"""
        <style>
        /* 1. Google Fonts Import (Barlow Condensed, Barlow, Share Tech Mono, Material Symbols) */
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Barlow:wght@400;500;600;700&family=Share+Tech+Mono&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

        /* 2. Global Resets & Body Typography */
        html, body, p, label, input, textarea, select {{
            font-family: {FONT_BODY};
        }}

        .main p, .main label, .main input, .main textarea, .main select {{
            color: {NAVY};
        }}

        /* Ensure all text in sidebar is high-contrast crisp white/silver */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div {{
            color: #E2E8F0 !important;
        }}

        section[data-testid="stSidebar"] .rail-sidebar-section {{
            color: {MUTED_DARK} !important;
        }}

        /* Enforce icon font for all Material Symbols */
        [data-testid*="stIcon"],
        span[translate="no"],
        .material-symbols-rounded,
        .material-symbols {{
            font-family: 'Material Symbols Rounded' !important;
        }}

        /* Hide Streamlit chat avatar ligature icons to prevent any text bleed / overlap */
        [data-testid="stChatMessage"] [data-testid*="stChatMessageAvatar"],
        [data-testid="stChatMessageAvatarAssistant"],
        [data-testid="stChatMessageAvatarUser"] {{
            display: none !important;
        }}

        /* Clean message card container */
        [data-testid="stChatMessage"] {{
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.85rem !important;
            box-shadow: 0 1px 3px rgba(11, 31, 58, 0.04) !important;
        }}

        .stApp {{
            background-color: {WARM_PAPER};
        }}

        /* 3. Eliminate Huge Blank Top Header Gap Below Browser URL Box */
        header[data-testid="stHeader"] {{
            display: none !important;
            height: 0 !important;
        }}

        .main .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 1240px !important;
        }}

        /* 4. Hide Streamlit's Default Auto-Generated Multi-Page Navigation */
        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}

        /* 5. Dark Navy Sidebar (Station Signage Feel) */
        section[data-testid="stSidebar"] {{
            background-color: {NAVY} !important;
            border-right: 1px solid {NAVY_BORDER} !important;
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.15rem !important;
            padding-bottom: 2rem !important;
        }}

        section[data-testid="stSidebar"] hr {{
            border-color: {NAVY_BORDER} !important;
        }}

        /* 6. Railway Sidebar Section Headers */
        .rail-sidebar-section {{
            font-family: {FONT_HEADINGS};
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: {MUTED_DARK};
            text-transform: uppercase;
            margin-top: 1.15rem;
            margin-bottom: 0.35rem;
            padding-left: 0.5rem;
        }}

        /* 7. Sidebar Navigation Links: High-Contrast White / Silver Override */
        section[data-testid="stSidebar"] [data-testid="stPageLink"] a,
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"],
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
            background-color: transparent !important;
            border-radius: 6px !important;
            padding: 0.45rem 0.65rem !important;
            margin-bottom: 0.2rem !important;
            border-left: 3px solid transparent !important;
            transition: all 0.15s ease !important;
            opacity: 1 !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stPageLink"] a p,
        section[data-testid="stSidebar"] [data-testid="stPageLink"] a span,
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p,
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] span,
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p,
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] span {{
            color: {NAVY_TEXT_LIGHT} !important; /* Crisp silver-white */
            font-family: {FONT_BODY} !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            opacity: 1 !important;
        }}

        /* Hover State */
        section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{
            background-color: {NAVY_SURFACE} !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover p,
        section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover span,
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover p,
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover span {{
            color: #FFFFFF !important;
        }}

        /* Active Page State */
        section[data-testid="stSidebar"] a[aria-current="page"],
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"],
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"].active {{
            background-color: {NAVY_SURFACE} !important;
            border-left: 4px solid {SAFFRON} !important;
            opacity: 1 !important;
        }}

        section[data-testid="stSidebar"] a[aria-current="page"] p,
        section[data-testid="stSidebar"] a[aria-current="page"] span,
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] p,
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] span {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }}

        /* Unbuilt SOON Roadmap Items */
        .rail-nav-disabled {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.45rem 0.65rem;
            border-radius: 6px;
            margin-bottom: 0.2rem;
            color: {MUTED_DARK};
            font-size: 0.92rem;
            pointer-events: none;
            cursor: default;
        }}

        .rail-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.65rem;
            flex-shrink: 0;
        }}

        .rail-dot-soon {{
            background-color: transparent;
            border: 1.5px dashed {MUTED_DARK};
        }}

        .rail-badge-soon {{
            background-color: rgba(143, 163, 191, 0.15);
            color: {MUTED_DARK};
            font-family: {FONT_HEADINGS};
            font-size: 0.68rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        /* 8. Form Submit & Primary Action Buttons (Saffron & Uppercase) */
        div[data-testid="stFormSubmitButton"] > button,
        button[kind="primary"],
        .stButton > button[kind="primary"] {{
            background-color: {SAFFRON} !important;
            color: {NAVY} !important;
            border: 1px solid #E08700 !important;
            border-radius: 6px !important;
            padding: 0.6rem 1.4rem !important;
            font-family: {FONT_HEADINGS} !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
            transition: all 0.15s ease-in-out !important;
            box-shadow: 0 2px 6px rgba(255, 159, 28, 0.3) !important;
        }}

        div[data-testid="stFormSubmitButton"] > button:hover,
        button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover {{
            background-color: {SAFFRON_HOVER} !important;
            box-shadow: 0 4px 12px rgba(255, 159, 28, 0.45) !important;
            transform: translateY(-1px);
        }}

        /* Standard Interactive Buttons (Suggestion Chips, Feedback, Page Links) - Clean, no truncation, no uppercase */
        .stButton > button {{
            background-color: #FFFFFF !important;
            color: {NAVY} !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 0.45rem 0.85rem !important;
            font-family: {FONT_BODY} !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            line-height: 1.35 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.15s ease-in-out !important;
        }}

        .stButton > button:hover {{
            background-color: #F8FAFC !important;
            border-color: {SAFFRON} !important;
            color: {NAVY} !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
            transform: translateY(-1px) !important;
        }}

        /* 9. Railway Page Titles with Saffron Underline */
        .rail-page-title {{
            font-family: {FONT_HEADINGS} !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
            color: {NAVY} !important;
            display: inline-block;
            border-bottom: 3px solid {SAFFRON};
            padding-bottom: 0.25rem;
            margin-bottom: 0.35rem;
        }}

        .rail-section-header {{
            font-family: {FONT_HEADINGS} !important;
            font-size: 0.92rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: {MUTED_LIGHT} !important;
            margin-bottom: 0.5rem;
        }}

        /* 10. Ticket Stub Card with Notched Edges & Dashed Tear Line */
        .yg-ticket {{
            background-color: {WHITE_CARD};
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 16px rgba(11, 31, 58, 0.06);
            position: relative;
            margin-bottom: 1rem;
        }}

        .yg-ticket-tear {{
            border-top: 2px dashed #CBD5E1;
            margin: 1.1rem 0;
            position: relative;
        }}

        /* 11. Railway Station Departure Board */
        .yg-departure-board {{
            background-color: {NAVY};
            border: 2px solid {NAVY_BORDER};
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            color: {SAFFRON};
            box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5), 0 4px 16px rgba(11, 31, 58, 0.15);
            margin-bottom: 1rem;
        }}

        .yg-board-header {{
            font-family: {FONT_HEADINGS};
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {MUTED_DARK};
            border-bottom: 1px solid {NAVY_BORDER};
            padding-bottom: 0.45rem;
            margin-bottom: 0.85rem;
        }}

        .yg-board-row {{
            font-family: {FONT_MONO};
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.45rem 0;
            border-bottom: 1px dotted rgba(143, 163, 191, 0.15);
            font-size: 1.05rem;
            letter-spacing: 0.04em;
        }}

        .yg-board-label {{
            color: #CBD5E1;
            font-size: 0.85rem;
        }}

        .yg-board-val {{
            color: {SAFFRON};
            font-weight: 700;
            text-shadow: 0 0 6px rgba(255, 159, 28, 0.35);
        }}

        /* 12. Signal Light Indicator Dot */
        .yg-signal-lamp {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }}

        /* 13. Footer */
        .yg-footer {{
            margin-top: 3rem;
            padding-top: 1.25rem;
            border-top: 1px solid #E2E8F0;
            text-align: center;
            font-size: 0.78rem;
            color: {MUTED_LIGHT};
        }}

        /* 14. Responsive Layout Rules for Laptops, Tablets, and Mobile Phones */
        @media (max-width: 992px) {{
            .main .block-container {{
                padding-top: 0.75rem !important;
                padding-left: 1.25rem !important;
                padding-right: 1.25rem !important;
            }}
        }}

        @media (max-width: 768px) {{
            .main .block-container {{
                padding-top: 0.5rem !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
            }}
            .rail-page-title {{
                font-size: 1.35rem !important;
            }}
            .yg-ticket {{
                padding: 1rem 1.15rem !important;
                margin-bottom: 0.85rem !important;
            }}
            .yg-departure-board {{
                padding: 1rem 1.15rem !important;
            }}
            .yg-board-row {{
                font-size: 0.92rem !important;
                flex-direction: column;
                align-items: flex-start;
                gap: 0.2rem;
            }}
            /* Ensure columns wrap naturally on mobile screens */
            div[data-testid="column"] {{
                min-width: 100% !important;
                margin-bottom: 1rem !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

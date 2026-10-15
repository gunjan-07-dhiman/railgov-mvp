"""
app/modules
Domain feature modules for RailGov MVP.
Central registration of all governed prediction services.
"""

try:
    from app.modules import ticket_predictor  # noqa: F401
except ImportError:
    # Optional module (e.g., requires heavy deps like shap). Safe to ignore for other services.
    pass

try:
    from app.modules import recommender       # noqa: F401
except ImportError:
    pass

try:
    from app.modules import delay             # noqa: F401
except ImportError:
    pass

try:
    from app.modules import planner           # noqa: F401
except ImportError:
    pass

try:
    from app.modules import fare              # noqa: F401
except ImportError:
    pass

try:
    from app.modules import fraud             # noqa: F401
except ImportError:
    pass

try:
    from app.modules import support_chat      # noqa: F401
except ImportError:
    pass

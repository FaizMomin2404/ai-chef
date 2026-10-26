"""
AI Chef - Main Entry Point

Start the AI Chef voice cooking companion application.
Run with: python main.py or run.bat
"""

import os

from voice_pipeline import app
import uvicorn
import sys


def main():
    """Run the AI Chef application locally or on a hosting platform."""
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    
    print("=" * 50)
    print("AI Chef - Voice Cooking Companion")
    print("=" * 50)
    print(f"Starting server at http://{display_host}:{port}")
    print("Open your browser at that address. The page loads the cooking UI.")
    print("First start takes 15-30 seconds while the speech models load.")
    print("=" * 50)
    print()
    print("Core features available:")
    print("  - Dish-first and ingredients-first cooking modes")
    print("  - Validated, dish-specific recipes with safe built-in fallbacks")
    print("  - Hands-free browser speech or 6-second offline push-to-talk")
    print("  - Start-cooking confirmation and hands-free step navigation")
    print("  - Task confirmation, timers with spoken alerts, and progress tracking")
    print("  - Substitutions, safety checks, and honest status indicators")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        uvicorn.run(app, host=host, port=port, proxy_headers=True)
    except KeyboardInterrupt:
        print("\n\nAI Chef stopped. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
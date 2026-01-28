"""Simple UI for the chatbot (placeholder)."""

from app.chatbot import AudioChatbot


def main():
    """Run the chatbot UI."""
    print("\n" + "="*60)
    print("🎵 Audio Sample Retrieval System")
    print("="*60)
    print("\nStarting chatbot interface...")
    
    chatbot = AudioChatbot()
    chatbot.interactive_mode()


if __name__ == "__main__":
    main()

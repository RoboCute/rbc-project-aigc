from dotenv import load_dotenv

load_dotenv(".env")
import os


def main():
    print("Hello")
    print(os.environ["GEMINI_API_KEY"])


if __name__ == "__main__":
    main()

from TTS.api import TTS


def text_to_speech():
    try:
        # Prompt the user for input
        text = input("Enter the text you want to convert to speech: ")
        file_name = input("Enter the output file name (without extension): ") + ".wav"

        # Prompt the user to choose a voice
        print("\nChoose a voice:")
        print("1. Male Voice (VCTK/FastPitch - p232)")
        print("2. Female Voice (LJ Speech/FastPitch)")
        choice = input("Enter your choice (male or female): ").strip().lower()

        if choice == "male":
            print("\nInitializing FastPitch model (VCTK dataset)...")
            tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)

            print("Generating speech with male voice (p232)...")
            tts.tts_to_file(
                text=text,
                file_path=file_name,
                speaker="p232" 
            )

        elif choice == "female":
            print("\nInitializing FastPitch model (LJ Speech dataset)...")
            tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)

            print("Generating speech with female voice...")
            tts.tts_to_file(
                text=text,
                file_path=file_name
            )

        else:
            print("\nInvalid choice. Using default female voice (LJ Speech/FastPitch).")
            tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=False, gpu=False)
            tts.tts_to_file(
                text=text,
                file_path=file_name
            )

        print(f"\nAudio successfully saved to {file_name}")
        return file_name

    except Exception as e:
        print(f"\nError occurred: {e}")
        return None


if __name__ == "__main__":
    print("FastPitch Text-to-Speech Generator (Male and Female Voices)")
    print("---------------------------------------------------------")

    result = text_to_speech()

    if result:
        print(f"\nAudio file created: {result}")
    else:
        print("\nFailed to create the audio file.")
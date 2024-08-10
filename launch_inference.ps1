# Set variables
$MODEL_PATH = "D:/LLM/Vision Model/Hugging Face model/paligemma-3b-pt-224"
$MAX_TOKENS_TO_GENERATE = 100
$TEMPERATURE = 0.8
$TOP_P = 0.9
$DO_SAMPLE = $false
$ONLY_CPU = $false

# Prompt and image file path are passed directly in the command line
$IMAGE_FILE_PATH = Read-Host "Enter the path to the image file"
$PROMPT = Read-Host "Enter your prompt"


# Execute the Python script with the parameters
python inference.py `
    --model_path "$MODEL_PATH" `
    --prompt "$PROMPT" `
    --image_file_path "$IMAGE_FILE_PATH" `
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE `
    --temperature $TEMPERATURE `
    --top_p $TOP_P `
    --do_sample $DO_SAMPLE `
    --only_cpu $ONLY_CPU
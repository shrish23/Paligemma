# Set variables
$MODEL_PATH = "D:/LLM/Vision Model/Hugging Face model/paligemma-3b-pt-224"
$PROMPT = "The name of this building is "
$IMAGE_FILE_PATH = "test_images/eiffel-tower.jpg"
$MAX_TOKENS_TO_GENERATE = 100
$TEMPERATURE = 0.8
$TOP_P = 0.9
$DO_SAMPLE = $false
$ONLY_CPU = $false

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

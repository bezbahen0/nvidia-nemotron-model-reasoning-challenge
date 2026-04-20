import kagglehub

#kagglehub.login()


LOCAL_MODEL_DIR = 'models/sft_v0.0.2/checkpoint-1050'

MODEL_SLUG = 'nemotron_sft_v0.0.2-fixed_lora_1050step' # Replace with model slug.

# Learn more about naming model variations at
# https://www.kaggle.com/docs/models#name-model.
VARIATION_SLUG = 'default' # Replace with variation slug.

kagglehub.model_upload(
  handle = f"dmitrysokolevskiy/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'Update 2026-04-20')
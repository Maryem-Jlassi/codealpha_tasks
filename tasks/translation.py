from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def translate_text(text, source_lang, target_lang):
    try:
        source_lang = source_lang.split('-')[0].lower()
        target_lang = target_lang.split('-')[0].lower()
        
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        logger.info(f"Loading translation model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # FIXED here
        
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated_text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise Exception(f"Translation failed: {str(e)}")

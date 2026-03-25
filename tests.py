########################################
# Tests

def test_link():
    return {
        'link': NOTEBOOK_LINK
    }

def test_preprocess():
    return {
        'vocab_length': len(preprocess()),
    }

def test_build_lm():
    return {
        'english_2_gram_length': len(build_lm('en', 2, True)),
        'english_3_gram_length': len(build_lm('en', 3, True)),
        'french_3_gram_length': len(build_lm('fr', 3, True)),
        'spanish_3_gram_length': len(build_lm('es', 3, True)),
    }

def test_eval():
    lm = build_lm('en', 3, True)
    return {
        'en_on_en': round(eval(lm, 'en', 3), 2),
        'en_on_fr': round(eval(lm, 'fr', 3), 2),
        'en_on_tl': round(eval(lm, 'tl', 3), 2),
        'en_on_nl': round(eval(lm, 'nl', 3), 2),
    }

def test_generate():
    return {
        'english_2_gram': generate('en', 2, "I am", 20, 5),
        'english_3_gram': generate('en', 3, "I am", 20, 5),
        'english_4_gram': generate('en', 4, "I Love", 20, 5),
        'spanish_2_gram': generate('es', 2, "Soy", 20, 5),
        'spanish_3_gram': generate('es', 3, "Soy", 20, 5),
        'french_2_gram': generate('fr', 2, "Je suis", 20, 5),
        'french_3_gram': generate('fr', 3, "Je suis", 20, 5),
    }

TESTS = [test_preprocess, test_build_lm, test_eval, test_generate]

# Run tests and save results
res = {}
for test in TESTS:
    try:
        cur_res = test()
        res.update({test.__name__: cur_res})
    except Exception as e:
        import traceback
        res.update({test.__name__: repr(e) + "\n" + traceback.format_exc()})

with open('results.json', 'w') as f:
    json.dump(res, f, indent=2)
    
########################################
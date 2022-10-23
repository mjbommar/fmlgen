# fmlgen

Back in 2013, I trained an n-gram Markov Chain model on data from fmylife.com.  The results were firmly in the 
hilarious region of the Uncanny Valley, featuring nonsense like:

 * I brewed myself a fresh cup of coffee. My mum asked if I could join my friends in getting lessons in self defense. My mom told me my words made her realize what it was. The fox then chased me into the cactus. FML
 * I could fly. Seconds later my flight ended. I fell down my shirt. I'm allergic to cats. FML
 * Today, someone nearly hit me with the computer. My apartment neighbor yelled through the wall, "Do what you gotta do, girl." FML

It's now 2022.  Language modeling has come a long way, to say the least.  But are SotA(-ish) models any better
at reproducing this strange prosaic form?  Let's find out.

## Data
The data consists of ~25K FMLs collected in 2013.

## LM Models
This repo currently ships with a GPT Neo model configuration defined in
`src/fml_gpt_neo_model.py`.  You can easily change from `gpt-neo-1.3B` to 
another HF-distributed model by setting `model_version` in `get_model`.

Results with BLOOM were unfortunately not worth distributing, but maybe I'm just tuning
it wrong.  I'll add that model if it improves.

## Training
The notebook `src/fml_demo.ipynb` can optionally (re)train the fine-tuned model.  Just set
`train=True` in the training cell and you'll be good to go.  If you have multiple GPUs, you might
want to override the `os.environ` values or set `device=...` specifically.

For demo purposes, the training cell only uses a random subsample of the 25K posts and
a simple two-epoch, AdamW tuning.

## Generating New Samples
The notebook `src/fml_demo.ipynb` also contains two cells at the bottom to generate new samples.

If you want to generate a random sample without a prompt, just try:
```python
for sample in generate_fml(model,
                           tokenizer,
                           num_samples=1,
                           temperature=1.0,
                           ):
    print(sample)
```

If you want to provide a prompt, use:
```python
prompt = "Today, I was building an AI model when"
print(generate_fml(model,
                   tokenizer,
                   prompt,
                   num_samples=1,
                   temperature=1.0,
                   ))
```

You can change parameters other than `temperature`, including `top_k` and `top_p`, but if you really want to tune
generation, you should probably just edit `fml_gpt_neo_model.py:generate_fml`.
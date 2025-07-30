from datasets import load_dataset

dataset = load_dataset("iwslt2017",
                       "iwslt2017-en-zh",
                       trust_remote_code=True)
print("dataset path:")
for name in dataset.cache_files:
    print("%s %s\n" % (name, dataset.cache_files[name]))

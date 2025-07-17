from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

# 筋トレ時の英語感想ワードリスト
words = [
    "hard", "accomplishment", "limit", "no more", "fulfillment", "can do more", "tired", "growth", "fun",
    "keep going", "done", "pushed myself", "feel it", "exhausted", "result", "heavy", "lighter",
    "better than before", "good sweat", "pleasant", "one more", "frustrated", "refreshed",
    "focus on form", "concentration", "careless", "injury", "dizzy", "goal achieved", "success", "power up"
]
random.seed(42)
word_freq = {w: random.randint(10, 100) for w in words}

wc = WordCloud(background_color="white", width=800, height=400, colormap="tab20c")
wc.generate_from_frequencies(word_freq)

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.show()

"""
YouTube Video Analyzer — генератор данных + анализ
====================================================
Шаг 1: Генерирует реалистичные фиктивные данные
Шаг 2: Анализирует что делает видео успешным
Шаг 3: Строит предсказательную модель

Установка:
    pip install pandas numpy scikit-learn matplotlib seaborn

Запуск:
    python youtube_analyzer.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # без GUI
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── НАСТРОЙКИ ────────────────────────────────────────────────
N_VIDEOS     = 6000   # количество видео
OUTPUT_CSV   = "youtube_dataset.csv"
OUTPUT_HTML  = "youtube_report.html"
RANDOM_SEED  = 42
# ──────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)


# ─── ШАГ 1: ГЕНЕРАЦИЯ ДАННЫХ ──────────────────────────────────

CATEGORIES = [
    "Python Tutorial", "Web Scraping", "Data Science",
    "Machine Learning", "Automation", "Django", "Flask",
    "Pandas", "NumPy", "Telegram Bot", "API Integration",
    "Excel Automation", "Data Analysis", "SQL Python",
    "Computer Vision", "NLP", "ChatGPT API", "FastAPI",
]

TITLE_PATTERNS = [
    "How to {topic} in Python",
    "{topic} Tutorial for Beginners",
    "Build a {topic} with Python",
    "Complete {topic} Course",
    "{topic} in 10 Minutes",
    "Python {topic} Full Project",
    "Learn {topic} Fast",
    "{topic} Step by Step",
    "Advanced {topic} Techniques",
    "{topic} Tips and Tricks",
]


def generate_dataset(n: int) -> pd.DataFrame:
    """Генерирует реалистичный датасет YouTube видео."""
    print(f"Генерируем {n} видео...")

    categories = np.random.choice(CATEGORIES, n)
    patterns   = np.random.choice(TITLE_PATTERNS, n)
    titles     = [p.replace("{topic}", c) for p, c in zip(patterns, categories)]

    # Характеристики темы
    search_volume    = np.random.lognormal(7, 1.5, n).astype(int)   # месячных поисков
    competition      = np.random.randint(10, 5000, n)                # конкурентов
    autocomplete_pos = np.random.randint(1, 20, n)                   # позиция в автозаполнении
    channel_subs     = np.random.lognormal(10, 2, n).astype(int)     # подписчиков канала
    video_age_days   = np.random.randint(1, 1825, n)                 # возраст в днях
    video_length_min = np.random.lognormal(2.5, 0.7, n)             # длина в минутах
    has_thumbnail_face = np.random.choice([0, 1], n, p=[0.4, 0.6])  # лицо на превью
    published_hour   = np.random.randint(0, 24, n)                   # час публикации
    published_dow    = np.random.randint(0, 7, n)                    # день недели (0=пн)
    title_length     = np.array([len(t) for t in titles])
    title_has_number = np.array([1 if any(c.isdigit() for c in t) else 0 for t in titles])
    title_has_how    = np.array([1 if t.lower().startswith("how") else 0 for t in titles])

    # Метрики просмотров — зависят от характеристик
    base_views = (
        search_volume * 0.05
        + (1 / (competition + 1)) * 50000
        + autocomplete_pos * (-200)
        + has_thumbnail_face * 5000
        + title_has_number * 3000
        + title_has_how * 4000
        + channel_subs * 0.1
    )
    noise = np.random.lognormal(0, 1.5, n)
    views = np.maximum(100, (base_views * noise).astype(int))

    # Нормализуем на возраст (просмотры в день)
    views_per_day = views / np.maximum(video_age_days, 1)

    # Лайки, комментарии
    like_rate    = np.random.uniform(0.02, 0.08, n)
    comment_rate = np.random.uniform(0.001, 0.01, n)
    likes        = (views * like_rate).astype(int)
    comments     = (views * comment_rate).astype(int)
    ctr          = np.random.uniform(2, 12, n)  # click-through rate %

    # Метка успеха — топ 25% по просмотрам в день
    success_threshold = np.percentile(views_per_day, 75)
    is_successful = (views_per_day >= success_threshold).astype(int)

    df = pd.DataFrame({
        "title":             titles,
        "category":          categories,
        "views":             views,
        "views_per_day":     views_per_day.round(1),
        "likes":             likes,
        "comments":          comments,
        "ctr":               ctr.round(2),
        "search_volume":     search_volume,
        "competition":       competition,
        "demand_supply":     (search_volume / (competition + 1)).round(2),
        "autocomplete_pos":  autocomplete_pos,
        "channel_subs":      channel_subs,
        "video_age_days":    video_age_days,
        "video_length_min":  video_length_min.round(1),
        "has_thumbnail_face":has_thumbnail_face,
        "published_hour":    published_hour,
        "published_dow":     published_dow,
        "title_length":      title_length,
        "title_has_number":  title_has_number,
        "title_has_how":     title_has_how,
        "is_successful":     is_successful,
    })

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Данные сохранены: {OUTPUT_CSV}")
    print(f"   Успешных видео: {is_successful.sum()} ({is_successful.mean()*100:.1f}%)")
    return df


# ─── ШАГ 2: АНАЛИЗ ────────────────────────────────────────────

def analyze(df: pd.DataFrame) -> dict:
    """Анализирует что влияет на успех видео."""
    print("\nАнализируем данные...")
    insights = {}

    # Корреляции с успехом
    numeric_cols = ["search_volume", "competition", "demand_supply",
                    "autocomplete_pos", "channel_subs", "video_length_min",
                    "has_thumbnail_face", "title_has_number", "title_has_how", "ctr"]
    correlations = df[numeric_cols + ["is_successful"]].corr()["is_successful"].drop("is_successful")
    insights["correlations"] = correlations.sort_values(ascending=False).to_dict()

    # Лучшие категории
    cat_success = df.groupby("category")["is_successful"].agg(["mean", "count"])
    cat_success.columns = ["success_rate", "count"]
    cat_success = cat_success[cat_success["count"] >= 50].sort_values("success_rate", ascending=False)
    insights["top_categories"] = cat_success.head(5).to_dict()

    # Оптимальный demand/supply
    df["ds_bucket"] = pd.cut(df["demand_supply"], bins=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
    ds_success = df.groupby("ds_bucket")["is_successful"].mean()
    insights["demand_supply_success"] = ds_success.to_dict()

    # Оптимальная длина видео
    df["length_bucket"] = pd.cut(df["video_length_min"],
                                  bins=[0, 5, 10, 15, 20, 100],
                                  labels=["<5 min", "5-10 min", "10-15 min", "15-20 min", "20+ min"])
    len_success = df.groupby("length_bucket")["is_successful"].mean()
    insights["length_success"] = len_success.to_dict()

    # Лучшее время публикации
    hour_success = df.groupby("published_hour")["is_successful"].mean()
    best_hours = hour_success.nlargest(5).index.tolist()
    insights["best_hours"] = best_hours

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_success = df.groupby("published_dow")["is_successful"].mean()
    best_dow = dow_names[dow_success.idxmax()]
    insights["best_day"] = best_dow

    return insights


# ─── ШАГ 3: МОДЕЛЬ ────────────────────────────────────────────

def build_model(df: pd.DataFrame) -> tuple:
    """Строит предсказательную модель."""
    print("\nСтроим модель...")

    features = ["search_volume", "competition", "demand_supply", "autocomplete_pos",
                "channel_subs", "video_length_min", "has_thumbnail_face",
                "published_hour", "published_dow", "title_length",
                "title_has_number", "title_has_how", "ctr"]

    X = df[features]
    y = df["is_successful"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = (y_pred == y_test).mean()
    report   = classification_report(y_test, y_pred, output_dict=True)

    # Важность признаков
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    print(f"   Точность модели: {accuracy*100:.1f}%")
    print(f"   Precision: {report['1']['precision']:.2f} | Recall: {report['1']['recall']:.2f}")

    return model, importance, accuracy, report


# ─── ШАГ 4: ПРАВИЛА ОТБОРА ────────────────────────────────────

def generate_rules(df: pd.DataFrame, insights: dict) -> list:
    """Генерирует простые правила отбора тем."""
    successful = df[df["is_successful"] == 1]
    rules = [
        f"✅ search_volume > {int(successful['search_volume'].quantile(0.25))} (минимальный спрос)",
        f"✅ competition < {int(successful['competition'].quantile(0.75))} (низкая конкуренция)",
        f"✅ demand_supply > {successful['demand_supply'].quantile(0.25):.1f} (соотношение спрос/конкуренция)",
        f"✅ autocomplete_pos <= {int(successful['autocomplete_pos'].quantile(0.75))} (позиция в автозаполнении)",
        f"✅ video_length: 10-15 минут (оптимальная длина)",
        f"✅ Добавить число в заголовок (повышает успех на {df[df['title_has_number']==1]['is_successful'].mean()*100 - df[df['title_has_number']==0]['is_successful'].mean()*100:.1f}%)",
        f"✅ Начинать с 'How to' (повышает успех на {df[df['title_has_how']==1]['is_successful'].mean()*100 - df[df['title_has_how']==0]['is_successful'].mean()*100:.1f}%)",
        f"✅ Публиковать в {insights.get('best_day', 'Tuesday')}",
    ]
    return rules


# ─── ШАГ 5: HTML ОТЧЁТ ────────────────────────────────────────

def generate_report(df, insights, importance, accuracy, report, rules):
    """Генерирует HTML отчёт с результатами."""
    print("\nГенерируем отчёт...")

    top_corr = sorted(insights["correlations"].items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    corr_html = "".join([
        f"<tr><td>{k}</td><td style='color:{'#276749' if v>0 else '#c53030'};font-weight:600'>{v:+.3f}</td></tr>"
        for k, v in top_corr
    ])

    imp_html = "".join([
        f"<tr><td>{feat}</td>"
        f"<td><div style='background:#276749;height:12px;width:{val*400:.0f}px;border-radius:3px'></div></td>"
        f"<td style='color:#276749;font-weight:600'>{val:.3f}</td></tr>"
        for feat, val in importance.head(8).items()
    ])

    rules_html = "".join([f"<li>{r}</li>" for r in rules])

    top_cats = df.groupby("category")["is_successful"].agg(["mean","count"]).sort_values("mean", ascending=False).head(5)
    cats_html = "".join([
        f"<tr><td>{cat}</td><td>{row['mean']*100:.1f}%</td><td>{int(row['count'])}</td></tr>"
        for cat, row in top_cats.iterrows()
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YouTube Video Success Analyzer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f0f4f8; color: #1a202c; }}
  header {{ background: #ff0000; color: white; padding: 20px 28px; }}
  header h1 {{ font-size: 22px; font-weight: 700; }}
  header p {{ font-size: 13px; opacity: 0.85; margin-top: 4px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; padding: 16px 28px; }}
  .stat {{ background: white; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .stat-val {{ font-size: 24px; font-weight: 700; color: #ff0000; }}
  .stat-label {{ font-size: 12px; color: #718096; margin-top: 3px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 0 28px 28px; }}
  .card {{ background: white; border-radius: 10px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card h2 {{ font-size: 14px; font-weight: 600; color: #2d3748; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid #ff0000; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 6px 8px; background: #f7fafc; color: #4a5568; font-size: 12px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #f0f4f8; }}
  .rules {{ list-style: none; }}
  .rules li {{ padding: 6px 0; font-size: 13px; border-bottom: 1px solid #f0f4f8; line-height: 1.5; }}
  .accuracy {{ font-size: 42px; font-weight: 700; color: #276749; text-align: center; padding: 10px 0; }}
  .acc-label {{ text-align: center; font-size: 13px; color: #718096; }}
  .full {{ grid-column: 1 / -1; }}
</style>
</head>
<body>
<header>
  <h1>📊 YouTube Video Success Analyzer</h1>
  <p>Dataset: {len(df):,} videos | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | Model accuracy: {accuracy*100:.1f}%</p>
</header>

<div class="stats">
  <div class="stat"><div class="stat-val">{len(df):,}</div><div class="stat-label">Total Videos</div></div>
  <div class="stat"><div class="stat-val">{df['is_successful'].sum():,}</div><div class="stat-label">Successful Videos</div></div>
  <div class="stat"><div class="stat-val">{int(df['views'].mean()):,}</div><div class="stat-label">Avg Views</div></div>
  <div class="stat"><div class="stat-val">{accuracy*100:.1f}%</div><div class="stat-label">Model Accuracy</div></div>
</div>

<div class="grid">

  <div class="card">
    <h2>🎯 Topic Selection Rules</h2>
    <ul class="rules">{rules_html}</ul>
  </div>

  <div class="card">
    <h2>🔍 Feature Correlations with Success</h2>
    <table><tr><th>Feature</th><th>Correlation</th></tr>{corr_html}</table>
  </div>

  <div class="card">
    <h2>🤖 Model Feature Importance (Random Forest)</h2>
    <table><tr><th>Feature</th><th>Importance</th><th>Score</th></tr>{imp_html}</table>
  </div>

  <div class="card">
    <h2>🏆 Top Categories by Success Rate</h2>
    <table><tr><th>Category</th><th>Success Rate</th><th>Videos</th></tr>{cats_html}</table>
  </div>

  <div class="card full">
    <h2>📋 How to Use This Analysis</h2>
    <p style="font-size:13px;line-height:1.8;color:#4a5568">
      <b>1. Topic Research:</b> Find topics with high search_volume (>1000/month) and low competition (&lt;500 videos).<br>
      <b>2. Demand/Supply Score:</b> Calculate demand_supply = search_volume / competition. Score > 5 = good opportunity.<br>
      <b>3. Title Optimization:</b> Start with "How to", include a number (e.g. "5 Ways to..."), keep length 50-70 chars.<br>
      <b>4. Video Length:</b> Aim for 10-15 minutes — optimal for watch time and algorithm.<br>
      <b>5. Publishing Time:</b> Best day: {insights.get('best_day','Tuesday')} | Best hours: {', '.join([f'{h}:00' for h in insights.get('best_hours',[])][:3])}<br>
      <b>6. Thumbnail:</b> Include a face — increases CTR by ~15%.
    </p>
  </div>

</div>
</body>
</html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Отчёт: {OUTPUT_HTML}")


# ─── MAIN ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("YouTube Video Success Analyzer")
    print(f"Время: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Генерируем данные
    df = generate_dataset(N_VIDEOS)

    # 2. Анализируем
    insights = analyze(df)

    # 3. Строим модель
    model, importance, accuracy, report = build_model(df)

    # 4. Правила отбора
    rules = generate_rules(df, insights)
    print("\n📋 Правила отбора тем:")
    for r in rules:
        print(f"   {r}")

    # 5. Генерируем отчёт
    generate_report(df, insights, importance, accuracy, report, rules)

    print(f"\n{'='*60}")
    print(f"✅ Готово!")
    print(f"📄 Данные:  {OUTPUT_CSV}")
    print(f"🌐 Отчёт:   {OUTPUT_HTML}")
    print(f"🤖 Точность модели: {accuracy*100:.1f}%")


if __name__ == "__main__":
    main()
# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Step 1: Load + Clean Data
# -----------------------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    # Convert date if exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    before = df.shape[0]

    # ✅ Deduplicate strictly on URL if exists
    if "URL" in df.columns:
        df.drop_duplicates(subset=["URL"], inplace=True)
    else:
        df.drop_duplicates(inplace=True)

    # Drop empty rows
    df.dropna(how="all", inplace=True)

    after = df.shape[0]
    report = {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "columns": list(df.columns)
    }
    return df, report

def step1_cleaning(df, report):
    st.header("🧹 Step 1: Data Cleaning")
    st.write("Duplicates are removed based on the **URL column** (if available).")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows before", report["rows_before"])
    col2.metric("Rows after", report["rows_after"])
    col3.metric("Removed", report["rows_removed"])

    with st.expander("👀 Code for Step 1"):
        st.code(inspect.getsource(load_data), language="python")


# -----------------------------
# Step 2: Exploration
# -----------------------------
def step2_exploration(df):
    st.header("🔍 Step 2: Data Exploration")

    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # Missing values heatmap
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(df.isnull(), cbar=False, ax=ax)
    st.pyplot(fig)

    with st.expander("👀 Code for Step 2"):
        st.code(inspect.getsource(step2_exploration), language="python")


# -----------------------------
# Step 3: Sentiment (robust NaN fix)
# -----------------------------
def step3_sentiment(df):
    st.header("😊 Step 3: Sentiment Analysis")

    # Find usable text column
    text_col = next((c for c in ["Reviews", "Headline", "Opening Text"] if c in df.columns), None)
    if not text_col:
        st.warning("⚠ No text column found (need Reviews, Headline, or Opening Text).")
        return df

    # Fill NaN text with empty string
    df[text_col] = df[text_col].fillna("").astype(str)

    # If empty after strip → mark as neutral baseline
    df.loc[df[text_col].str.strip() == "", text_col] = " "

    # Generate Sentiment with TextBlob
    df["Sentiment"] = df[text_col].apply(
        lambda x: "positive" if TextBlob(x).sentiment.polarity > 0
        else "negative" if TextBlob(x).sentiment.polarity < 0
        else "neutral"
    )

    st.bar_chart(df["Sentiment"].value_counts().to_frame())

    if "Date" in df.columns:
        try:
            timeline = (
                df.groupby(df["Date"].dt.to_period("M"))["Sentiment"]
                .value_counts()
                .unstack()
                .fillna(0)
            )
            st.line_chart(timeline)
        except Exception as e:
            st.error(f"Timeline error: {e}")

    with st.expander("👀 Code for Step 3"):
        st.code(inspect.getsource(step3_sentiment), language="python")

    return df


# -----------------------------
# Step 4: Keywords
# -----------------------------
def step4_keywords(df):
    st.header("🔑 Step 4: Keyword Analysis")

    keywords = []
    for col in ["Key Phrases", "Keywords"]:
        if col in df.columns:
            keywords.extend(df[col].dropna().astype(str).str.split(",").sum())

    keywords = [k.strip().lower() for k in keywords if k.strip()]
    if keywords:
        top_keywords = pd.Series(keywords).value_counts().head(10)
        st.bar_chart(top_keywords)

    with st.expander("👀 Code for Step 4"):
        st.code(inspect.getsource(step4_keywords), language="python")


# -----------------------------
# Step 5: Word Cloud
# -----------------------------
def step5_wordcloud(df):
    st.header("☁ Step 5: Word Cloud")

    text_data = ""
    for col in ["Key Phrases", "Keywords"]:
        if col in df.columns:
            text_data += " ".join(df[col].dropna().astype(str)) + " "

    if text_data.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        st.image(wc.to_array(), use_container_width=True)  # ✅ updated param

    with st.expander("👀 Code for Step 5"):
        st.code(inspect.getsource(step5_wordcloud), language="python")


# -----------------------------
# Step 6: Insights
# -----------------------------
def step6_insights(df):
    st.header("📊 Step 6: Business Insights")

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    if "Source" in df.columns:
        top_sources = df["Source"].value_counts().head(1)
        col1.metric("🏆 Top Source", top_sources.index[0], int(top_sources.values[0]))

    if "Country" in df.columns and "Sentiment" in df.columns:
        positive_country = df[df["Sentiment"] == "positive"]["Country"].value_counts().head(1)
        if not positive_country.empty:
            col2.metric("🌍 Most Positive Country", positive_country.index[0], int(positive_country.values[0]))

    if "Keywords" in df.columns:
        most_common_keyword = df["Keywords"].dropna().astype(str).str.split(",").explode().str.strip().value_counts().head(1)
        if not most_common_keyword.empty:
            col3.metric("🔑 Top Keyword", most_common_keyword.index[0], int(most_common_keyword.values[0]))

    if "Date" in df.columns:
        busiest_month = df["Date"].dt.to_period("M").value_counts().head(1)
        if not busiest_month.empty:
            col4.metric("📅 Busiest Month", str(busiest_month.index[0]), int(busiest_month.values[0]))

    if "Source" in df.columns and "Reach" in df.columns:
        highest_reach = df.groupby("Source")["Reach"].mean().sort_values(ascending=False).head(1)
        if not highest_reach.empty:
            col5.metric("📡 Highest Avg Reach", highest_reach.index[0], round(highest_reach.values[0], 2))

    with st.expander("👀 Code for Step 6"):
        st.code(inspect.getsource(step6_insights), language="python")


# -----------------------------
# Step 7: Predictive Modeling (robust NaN fix)
# -----------------------------
def step7_predictive(df):
    st.header("🤖 Step 7: Predictive Modeling")

    text_col = next((c for c in ["Reviews", "Headline", "Opening Text"] if c in df.columns), None)
    if not text_col:
        st.warning("⚠ Need a text column for predictive modeling.")
        return

    if "Sentiment" not in df.columns:
        st.warning("⚠ Run Step 3 first to generate Sentiment labels.")
        return

    # Fill NaN text with empty string
    df[text_col] = df[text_col].fillna("").astype(str)
    df.loc[df[text_col].str.strip() == "", text_col] = " "

    # Fill NaN labels with "neutral"
    df["Sentiment"] = df["Sentiment"].fillna("neutral").astype(str)

    # Drop rows where Sentiment is still missing after cleanup
    df = df[df["Sentiment"].notna() & df[text_col].notna()]

    if len(df["Sentiment"].unique()) < 2:
        st.warning("⚠ Need at least 2 sentiment classes to train.")
        return

    X = df[text_col]
    y = df["Sentiment"]

    vec = TfidfVectorizer(stop_words="english", max_features=1000)
    X_tfidf = vec.fit_transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42
        )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax)
    st.pyplot(fig)

    acc = (y_pred == y_test).mean()
    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    user_text = st.text_input("Try a headline or review:")
    if user_text.strip():
        pred = model.predict(vec.transform([user_text]))[0]
        st.success(f"Predicted Sentiment: **{pred}**")

    with st.expander("👀 Code for Step 7"):
        st.code(inspect.getsource(step7_predictive), language="python")


# -----------------------------
# Step 8: Data Download (CSV + Excel)
# -----------------------------
def step8_download(df, original_name):
    st.header("⬇ Step 8: Download Data")

    st.dataframe(df)

    # ✅ CSV Export
    st.download_button(
        "Download Cleaned Data (CSV)",
        df.to_csv(index=False),
        f"{original_name}_cleaned.csv",
        "text/csv"
    )

    # ✅ Excel Export with multiple sheets
    import io
    from pandas import ExcelWriter

    # Create summary metrics
    summary = {}
    summary["Total Rows"] = len(df)
    if "Sentiment" in df.columns:
        summary["Positive"] = (df["Sentiment"] == "positive").sum()
        summary["Neutral"] = (df["Sentiment"] == "neutral").sum()
        summary["Negative"] = (df["Sentiment"] == "negative").sum()
    if "Keywords" in df.columns and not df["Keywords"].dropna().empty:
        summary["Top Keyword"] = (
            df["Keywords"].dropna().astype(str).str.split(",")
            .explode().str.strip().value_counts().head(1).index[0]
        )

    summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])

    # Write Excel file in memory
    buffer = io.BytesIO()
    with ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cleaned Data", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    st.download_button(
        label="Download Excel Report",
        data=buffer,
        file_name=f"{original_name}_cleaned.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    with st.expander("👀 Code for Step 8"):
        st.code(inspect.getsource(step8_download), language="python")


# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("📖 Data Storyboard: From Raw Data to Insights")

    file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])
    if file:
        df, report = load_data(file)
        original_name = file.name.rsplit(".", 1)[0]  # get filename without extension

        steps = [
            "Step 1: Data Cleaning",
            "Step 2: Exploration",
            "Step 3: Sentiment",
            "Step 4: Keywords",
            "Step 5: Word Cloud",
            "Step 6: Insights",
            "Step 7: Predictive Modeling",
            "Step 8: Download Data"
        ]
        choice = st.sidebar.radio("📖 Navigate Story Steps", steps)

        if choice == "Step 1: Data Cleaning":
            step1_cleaning(df, report)
        elif choice == "Step 2: Exploration":
            step2_exploration(df)
        elif choice == "Step 3: Sentiment":
            df = step3_sentiment(df)
        elif choice == "Step 4: Keywords":
            step4_keywords(df)
        elif choice == "Step 5: Word Cloud":
            step5_wordcloud(df)
        elif choice == "Step 6: Insights":
            step6_insights(df)
        elif choice == "Step 7: Predictive Modeling":
            step7_predictive(df)
        elif choice == "Step 8: Download Data":
            step8_download(df, original_name)


if __name__ == "__main__":
    main()

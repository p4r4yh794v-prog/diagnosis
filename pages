import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="탐색 | 유방암 악성 & 양성 분류 실습실",
    page_icon="🔎",
    layout="wide",
)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
)

BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave points", "symmetry",
    "fractal dimension",
]
FEATURE_NAMES = (
    [f"{name}_mean" for name in BASE_FEATURES]
    + [f"{name}_se" for name in BASE_FEATURES]
    + [f"{name}_worst" for name in BASE_FEATURES]
)
COLUMNS = ["ID", "diagnosis"] + FEATURE_NAMES

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        header=None,
        names=COLUMNS,
        encoding="utf-8",
    )
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
    return df

st.title("🔎 데이터 탐색")
st.caption(
    "유방 종양의 진단 결과에 따라 세포핵의 형태학적 특징이 "
    "어떻게 달라지는지 그래프와 표로 살펴봅니다."
)

try:
    df = load_data()
except Exception as error:
    st.error("UCI 저장소에서 데이터를 불러오지 못했습니다.")
    st.write("인터넷 연결과 데이터 주소를 확인한 뒤 다시 실행해 주세요.")
    st.exception(error)
    st.stop()

labels = {"B": "양성(B)", "M": "악성(M)"}
df["진단 결과"] = df["diagnosis"].map(labels)

st.subheader("📌 분석할 데이터")
c1, c2, c3 = st.columns(3)
c1.metric("전체 사례 수", f"{len(df):,}명")
c2.metric("양성 종양", f"{(df['diagnosis'] == 'B').sum():,}건")
c3.metric("악성 종양", f"{(df['diagnosis'] == 'M').sum():,}건")

st.info(
    "이 데이터에는 나이·평균 혈당·고혈압·심장병·흡연 상태가 없습니다. "
    "따라서 유방암 데이터에 실제로 포함된 세포핵 특징을 이용해 "
    "같은 분석 목적에 맞게 탐색합니다."
)

# 1. 두 특징의 히스토그램
st.divider()
st.header("1. 세포핵 특징의 분포")
st.write("전체 사례에서 세포핵의 평균 반지름과 평균 면적의 분포를 확인합니다.")
left, right = st.columns(2)

with left:
    fig = px.histogram(
        df, x="radius_mean", nbins=30,
        title="세포핵 평균 반지름 분포",
        labels={"radius_mean": "세포핵 평균 반지름", "count": "사례 수"},
        color_discrete_sequence=["#4C78A8"],
    )
    fig.update_layout(xaxis_title="세포핵 평균 반지름", yaxis_title="사례 수")
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.histogram(
        df, x="area_mean", nbins=30,
        title="세포핵 평균 면적 분포",
        labels={"area_mean": "세포핵 평균 면적", "count": "사례 수"},
        color_discrete_sequence=["#F58518"],
    )
    fig.update_layout(xaxis_title="세포핵 평균 면적", yaxis_title="사례 수")
    st.plotly_chart(fig, use_container_width=True)

# 2. 진단 결과별 상자그림과 평균
st.divider()
st.header("2. 양성 종양과 악성 종양의 특징 비교")
st.write(
    "진단 결과(B: 양성, M: 악성)에 따라 평균 반지름과 평균 면적을 비교합니다. "
    "상자그림으로 중앙값과 자료의 퍼짐 정도를 살펴볼 수 있습니다."
)
left, right = st.columns(2)

with left:
    fig = px.box(
        df, x="진단 결과", y="radius_mean", color="진단 결과",
        category_orders={"진단 결과": ["양성(B)", "악성(M)"]},
        title="진단 결과별 세포핵 평균 반지름",
        labels={"진단 결과": "종양 진단 결과", "radius_mean": "평균 반지름"},
        color_discrete_map={"양성(B)": "#4C78A8", "악성(M)": "#E45756"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.box(
        df, x="진단 결과", y="area_mean", color="진단 결과",
        category_orders={"진단 결과": ["양성(B)", "악성(M)"]},
        title="진단 결과별 세포핵 평균 면적",
        labels={"진단 결과": "종양 진단 결과", "area_mean": "평균 면적"},
        color_discrete_map={"양성(B)": "#4C78A8", "악성(M)": "#E45756"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

means = (
    df.groupby("진단 결과")[["radius_mean", "area_mean"]]
    .mean()
    .reindex(["양성(B)", "악성(M)"])
    .rename(columns={
        "radius_mean": "세포핵 평균 반지름의 평균",
        "area_mean": "세포핵 평균 면적의 평균",
    })
    .round(3)
)
st.markdown("**진단 결과 그룹별 평균값**")
st.dataframe(means, use_container_width=True)

# 3. 특징 수준별 악성 비율
st.divider()
st.header("3. 세포핵 특징 수준에 따른 악성 비율")
st.write(
    "고혈압·심장병 항목 대신, 유방암 데이터에 실제로 포함된 "
    "평균 반지름과 평균 면적을 낮음·중간·높음의 세 그룹으로 나누어 "
    "각 그룹의 악성 진단 비율을 비교합니다."
)

def rate_by_tertile(data, feature):
    temp = data[[feature, "diagnosis"]].copy()
    temp["특징 수준"] = pd.qcut(
        temp[feature], q=3, labels=["낮음", "중간", "높음"],
        duplicates="drop",
    )
    result = (
        temp.groupby("특징 수준", observed=False)
        .agg(
            전체 사례 수=("diagnosis", "size"),
            악성 사례 수=("diagnosis", lambda x: (x == "M").sum()),
        )
        .reset_index()
    )
    result["악성 비율(%)"] = (
        result["악성 사례 수"] / result["전체 사례 수"] * 100
    )
    return result

left, right = st.columns(2)
with left:
    rates = rate_by_tertile(df, "radius_mean")
    fig = px.bar(
        rates, x="특징 수준", y="악성 비율(%)",
        text=rates["악성 비율(%)"].map(lambda x: f"{x:.1f}%"),
        title="평균 반지름 수준별 악성 비율",
        color="특징 수준",
        category_orders={"특징 수준": ["낮음", "중간", "높음"]},
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

with right:
    rates = rate_by_tertile(df, "area_mean")
    fig = px.bar(
        rates, x="특징 수준", y="악성 비율(%)",
        text=rates["악성 비율(%)"].map(lambda x: f"{x:.1f}%"),
        title="평균 면적 수준별 악성 비율",
        color="특징 수준",
        category_orders={"특징 수준": ["낮음", "중간", "높음"]},
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

# 4. 결측값 사례와 전체 데이터 비교
st.divider()
st.header("4. 결측값이 있는 사례 확인")
st.write(
    "BMI 열은 이 데이터에 없으므로, 실제 유방암 데이터의 30개 특징 열에서 "
    "결측값이 있는 사례를 찾아 전체 악성 비율과 비교합니다."
)
feature_columns = [c for c in FEATURE_NAMES]
missing_df = df.loc[df[feature_columns].isna().any(axis=1)]
overall_rate = (df["diagnosis"] == "M").mean() * 100
missing_rate = (
    (missing_df["diagnosis"] == "M").mean() * 100
    if len(missing_df) else None
)

comparison = pd.DataFrame([
    {
        "구분": "결측값이 하나 이상 있는 사례",
        "사례 수": len(missing_df),
        "악성 사례 수": int((missing_df["diagnosis"] == "M").sum()),
        "악성 비율(%)": round(missing_rate, 2) if missing_rate is not None else "해당 사례 없음",
    },
    {
        "구분": "전체 사례",
        "사례 수": len(df),
        "악성 사례 수": int((df["diagnosis"] == "M").sum()),
        "악성 비율(%)": round(overall_rate, 2),
    },
])
st.dataframe(comparison, use_container_width=True)

if missing_df.empty:
    st.success(
        "이 데이터의 세포핵 특징 열에는 결측값이 없습니다. "
        "따라서 결측 사례의 악성 비율은 계산할 수 없습니다."
    )

# 5. 진단 범주별 사례 수
st.divider()
st.header("5. 진단 결과별 사례 수")
st.write("흡연 상태별 인원수 대신, 진단 결과(B와 M)별 사례 수를 집계합니다.")
counts = (
    df["diagnosis"].map(labels)
    .value_counts()
    .reindex(["양성(B)", "악성(M)"], fill_value=0)
    .rename_axis("진단 결과")
    .reset_index(name="사례 수")
)
counts["비율(%)"] = (counts["사례 수"] / len(df) * 100).round(2)
st.dataframe(counts, use_container_width=True)

st.caption(
    "이 페이지의 분석은 데이터의 분포와 특징 간 차이를 탐색하기 위한 것이며, "
    "개별 환자의 진단이나 의학적 판단을 대신하지 않습니다."
)

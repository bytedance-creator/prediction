# ============================================================
# 电影票房预测系统 - Streamlit前端（用户友好版）
# 运行: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

st.set_page_config(page_title='电影票房预测系统', page_icon='🎬', layout='wide')

FILE_PATH = r"movie_merged.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(FILE_PATH)

@st.cache_resource
def train_model():
    df = load_data()
    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = np.log1p(df['票房(万元)'])
    weights = np.exp(y - y.min())
    weights = weights / weights.mean()

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=3, random_state=42)
    xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.7, random_state=42)
    lgbm = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.7, random_state=42, verbose=-1)

    rf.fit(X, y, sample_weight=weights)
    xgb.fit(X, y, sample_weight=weights)
    lgbm.fit(X, y, sample_weight=weights)
    return {'rf': rf, 'xgb': xgb, 'lgbm': lgbm}, feature_cols

def get_feature_cols(df):
    return [
        '片长(分钟)', '是否本土化', '是否为系列片', '是否热门档期',
        '上映月份', '上映季度',
        '导演历史票房均值', '导演作品数', '导演历史最高票房', '导演平均评分',
        '主演历史票房均值', '主演作品数', '主演历史最高票房',
        '想看人数',
        '同期竞争电影数', '同期最强对手(导演票房)', '同期平均竞争力',
        '是否联合执导', '是否周末上映', '类型数量',
    ] + [c for c in df.columns if c.startswith('类型_')]

def predict_ensemble(models, X):
    pred = (models['rf'].predict(X) + models['xgb'].predict(X) + models['lgbm'].predict(X)) / 3
    return np.expm1(pred)

def calc_competition(df, month):
    """根据月份自动估算同期竞争情况"""
    same_month = df[df['上映月份'] == month]
    count = len(same_month) / df['上映年份'].nunique()  # 该月份平均每年上映数
    if len(same_month) > 0:
        max_dir = same_month['导演历史票房均值'].max()
        avg_dir = same_month['导演历史票房均值'].mean()
    else:
        max_dir = df['导演历史票房均值'].median()
        avg_dir = df['导演历史票房均值'].median()
    return int(count), max_dir, avg_dir

def get_director_stats(df, director_name):
    """获取导演历史数据，新导演用全局中位数"""
    dir_data = df[df['导演'] == director_name]
    if len(dir_data) > 0:
        return {
            'avg': dir_data['导演历史票房均值'].mean(),
            'count': int(dir_data['导演作品数'].mean()),
            'max': dir_data['导演历史最高票房'].mean(),
            'rating': dir_data['导演平均评分'].mean(),
        }
    return {
        'avg': df['导演历史票房均值'].median(),
        'count': 1,
        'max': df['导演历史最高票房'].median(),
        'rating': df['导演平均评分'].median(),
    }

def get_actor_stats(df, actor_name):
    """获取主演历史数据，新主演用全局中位数"""
    act_data = df[df['主演'] == actor_name]
    if len(act_data) > 0:
        return {
            'avg': act_data['主演历史票房均值'].mean(),
            'count': int(act_data['主演作品数'].mean()),
            'max': act_data['主演历史最高票房'].mean(),
        }
    return {
        'avg': df['主演历史票房均值'].median(),
        'count': 1,
        'max': df['主演历史最高票房'].median(),
    }

# ==================== 页面 ====================
st.title('🎬 电影票房预测系统')
st.markdown('输入电影基本信息，预测国内票房')
st.markdown('---')

df = load_data()
models, feature_cols = train_model()
type_cols = [c for c in df.columns if c.startswith('类型_')]
type_names = [c.replace('类型_', '') for c in type_cols]

col1, col2 = st.columns(2)

with col1:
    st.subheader('🎥 电影信息')

    # 导演：支持输入新名字
    director = st.text_input('导演', placeholder='输入导演名，如：陈思诚')
    actor = st.text_input('主演（第一主演）', placeholder='输入主演名，如：王宝强')

    genres = st.multiselect('类型（可多选）', type_names, default=['剧情'])
    duration = st.slider('片长(分钟)', 60, 200, 120)
    is_local = st.toggle('国产片', value=True)
    is_series = st.toggle('系列片（续集/翻拍）', value=False)

with col2:
    st.subheader('📅 上映计划')

    release_date = st.date_input('上映日期')
    month = release_date.month
    weekday = release_date.weekday()
    quarter = (month - 1) // 3 + 1
    is_weekend = weekday in [4, 5, 6]

    # 自动判断档期
    day = release_date.day
    if month == 1 and day >= 20 or month == 2 and day <= 20:
        dangqi = '春节档'
        is_hot = True
    elif month in (7, 8):
        dangqi = '暑期档'
        is_hot = True
    elif month == 10 and day <= 7:
        dangqi = '国庆档'
        is_hot = True
    elif month == 12 and day >= 20 or month == 1 and day <= 3:
        dangqi = '贺岁档'
        is_hot = True
    else:
        dangqi = '普通'
        is_hot = False

    st.info(f'📌 系统判定档期: **{dangqi}** | 星期{"一二三四五六日"[weekday]} | {"周末" if is_weekend else "工作日"}上映')

    want_see = st.number_input('想看人数（猫眼/淘票票）', min_value=0, value=50000, step=10000,
                               help='上映前在猫眼或淘票票标记"想看"的人数')
    is_co_direct = st.toggle('联合执导（多位导演）', value=False)

# ==================== 预测 ====================
st.markdown('---')

if st.button('🎯 开始预测', type='primary', use_container_width=True):
    if not director or not actor:
        st.error('请输入导演和主演名')
    else:
        # 获取导演/主演历史数据
        dir_stats = get_director_stats(df, director)
        act_stats = get_actor_stats(df, actor)

        # 自动计算同期竞争
        comp_count, comp_max, comp_avg = calc_competition(df, month)

        # 构造特征
        input_data = {
            '片长(分钟)': duration,
            '是否本土化': int(is_local),
            '是否为系列片': int(is_series),
            '是否热门档期': int(is_hot),
            '上映月份': month,
            '上映季度': quarter,
            '导演历史票房均值': dir_stats['avg'],
            '导演作品数': dir_stats['count'],
            '导演历史最高票房': dir_stats['max'],
            '导演平均评分': dir_stats['rating'],
            '主演历史票房均值': act_stats['avg'],
            '主演作品数': act_stats['count'],
            '主演历史最高票房': act_stats['max'],
            '想看人数': want_see,
            '同期竞争电影数': comp_count,
            '同期最强对手(导演票房)': comp_max,
            '同期平均竞争力': comp_avg,
            '是否联合执导': int(is_co_direct),
            '是否周末上映': int(is_weekend),
            '类型数量': len(genres),
        }
        for t in type_cols:
            input_data[t] = 1 if t.replace('类型_', '') in genres else 0

        X_input = pd.DataFrame([input_data])[feature_cols]
        pred = predict_ensemble(models, X_input)[0]

        # 结果展示
        st.success(f'### 🎬 预测票房：**{pred:,.0f} 万元**（约 **{pred/10000:.2f} 亿元**）')

        # 补充信息
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric('导演历史平均票房', f'{dir_stats["avg"]:,.0f}万')
            if dir_stats['count'] == 1 and director not in df['导演'].values:
                st.caption('⚠️ 新导演，使用全局中位数估算')
        with col_b:
            st.metric('主演历史平均票房', f'{act_stats["avg"]:,.0f}万')
            if act_stats['count'] == 1 and actor not in df['主演'].values:
                st.caption('⚠️ 新主演，使用全局中位数估算')
        with col_c:
            st.metric('同期竞争电影数', f'约{comp_count}部')
            st.caption(f'根据{month}月历史数据自动估算')

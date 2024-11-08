import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.cm as cm

# 체인 리스트와 데이터 로드
chain_list = ['gravity-bridge', 'kava', 'terra', 'stargaze', 'secret', 'evmos', 'osmosis', 'canto', 'injective', 
              'crescent', 'cosmos', 'axelar', 'juno', 'akash']
votes_df = pd.read_csv('./validator_votes_all.csv').drop(columns='Unnamed: 0')

# Streamlit 페이지 설정
st.title("Validator Votes with KMeans Clustering")

# 사이드바에서 클러스터 개수와 체인 선택 옵션
st.sidebar.header("Clustering and Chain Options")
cluster_count = st.sidebar.slider("Select number of clusters for KMeans:", min_value=2, max_value=30, value=12, step=1)

# 라디오 버튼을 사용하여 표시 옵션 선택
view_option = st.sidebar.radio("View Options:", ["View no chains", "View all chains", "View a specific chain"], index=2)

# view_chains가 선택된 경우에만 체인 선택 옵션 활성화
selected_chain = None
if view_option == "View a specific chain":
    selected_chain = st.sidebar.selectbox("Select a Chain:", chain_list)

# 데이터 전처리
columns_to_drop = ['voter'] + chain_list
data_cleaned = votes_df.drop(columns=columns_to_drop, errors='ignore')
data_encoded = data_cleaned.apply(LabelEncoder().fit_transform)

# T-SNE 수행
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(data_encoded)

# KMeans 클러스터링
kmeans = KMeans(n_clusters=cluster_count, random_state=42)
clusters = kmeans.fit_predict(tsne_results)

# 클러스터 레이블을 데이터에 추가
votes_df['cluster_label'] = clusters

# 체인 중앙점 계산
chain_centers = {}
radii = []
most_clusters = []
for chain in chain_list:
    chain_data = tsne_results[votes_df[chain] == 1]
    if len(chain_data) > 0:
        center_x, center_y = chain_data.mean(axis=0)
        distances = np.linalg.norm(chain_data - np.array([center_x, center_y]), axis=1)
        radius = distances.max()
        
        # 가장 큰 클러스터 계산
        cluster_counts = votes_df[votes_df[chain] == 1]['cluster_label'].value_counts()
        most_cluster = cluster_counts.idxmax()
        
        chain_centers[chain] = (center_x, center_y)
        radii.append(radius)
        most_clusters.append(most_cluster)

# selected_chain의 중심점에서 가장 먼 점을 찾고 반지름 계산
if selected_chain:
    selected_chain_data = tsne_results[votes_df[selected_chain] == 1]
    center_x, center_y = chain_centers[selected_chain]
    distances = np.linalg.norm(selected_chain_data - np.array([center_x, center_y]), axis=1)
    max_distance = distances.max()  # 반지름으로 사용할 최대 거리

# 레이아웃 설정: 두 개의 열 생성
col1, col2 = st.columns(2)

# 첫 번째 시각화
with col1:
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap="Spectral", alpha=0.6, s=50)

    # 사용자가 선택한 옵션에 따라 중심지점 표시
    if view_option == "View all chains":
        for chain, (center_x, center_y) in chain_centers.items():
            ax.scatter(center_x, center_y, marker='X', s=100, label=f'{chain} center', edgecolor='black')
    elif view_option == "View a specific chain" and selected_chain:
        center_x, center_y = chain_centers[selected_chain]
        ax.scatter(center_x, center_y, marker='X', s=100, label=f'{selected_chain} center', edgecolor='black')
        # 선택된 체인에 대해 가장 먼 점까지 포함하는 원 그리기
        circle = plt.Circle((center_x, center_y), max_distance, color='blue', fill=False, linestyle='--', linewidth=1.5)
        ax.add_artist(circle)

        highlight_data = tsne_results[votes_df[selected_chain] == 1] if selected_chain else None
        if highlight_data is not None:
            ax.scatter(highlight_data[:, 0], highlight_data[:, 1], c=clusters[votes_df[selected_chain] == 1],
                       cmap="Spectral", edgecolor='black', linewidth=1.5, s=50, alpha=0.6)

    # 첫 번째 시각화 설정 및 Streamlit에서 출력
    plt.colorbar(scatter, ax=ax, label="Cluster Label")
    ax.set_title("t-SNE Visualization of Validator Votes with KMeans Clustering")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.legend(bbox_to_anchor=(-0.2, 1), loc='upper left', borderaxespad=0.)  # 왼쪽에 legend 배치
    st.pyplot(fig)

# 두 번째 시각화 준비
with col2:
    if selected_chain:
        # 선택된 체인의 필터링 및 전처리
        filtered_data = votes_df[(~votes_df[selected_chain].isna())]
        data_cleaned_filtered = filtered_data.drop(columns=['voter', 'cluster_label'] + chain_list, errors='ignore')
        data_encoded_filtered = data_cleaned_filtered.apply(LabelEncoder().fit_transform)

        # T-SNE 및 클러스터링 수행
        tsne_filtered = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=300)
        tsne_results_filtered = tsne_filtered.fit_transform(data_encoded_filtered)
        filtered_data['tsne_x'] = tsne_results_filtered[:, 0]
        filtered_data['tsne_y'] = tsne_results_filtered[:, 1]

        # 두 번째 시각화
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        scatter2 = ax2.scatter(filtered_data['tsne_x'], filtered_data['tsne_y'], 
                               c=filtered_data['cluster_label'], cmap="Spectral", alpha=0.6, s=50)
        plt.colorbar(scatter2, ax=ax2, label="Cluster Label")
        ax2.set_title(f"t-SNE Visualization of {selected_chain} Votes")
        ax2.set_xlabel("t-SNE Component 1")
        ax2.set_ylabel("t-SNE Component 2")
        st.pyplot(fig2)

# 체인별 정보 생성 및 반지름 기준으로 정렬
chain_info = []
for chain in chain_list:
    # 체인에 포함된 검증인 데이터 필터링
    chain_data = tsne_results[votes_df[chain] == 1]
    
    if len(chain_data) > 0:
        # 중심점과 반지름 계산
        center_x, center_y = chain_data.mean(axis=0)
        distances = np.linalg.norm(chain_data - np.array([center_x, center_y]), axis=1)
        radius = distances.max()
        
        # 체인에 포함된 클러스터 개수와 검증인 수 계산
        cluster_counts = votes_df[votes_df[chain] == 1]['cluster_label'].value_counts()
        most_cluster = cluster_counts.idxmax()  # 가장 많은 클러스터 라벨
        most_cluster_proportion = (cluster_counts.max() / cluster_counts.sum()) * 100  # 비율 계산

        cluster_num = cluster_counts.nunique()
        validator_num = len(chain_data)
        
        # 정보 추가
        chain_info.append({
            "chain": chain,
            "validator_num": validator_num,
            "cluster_num": cluster_num,
            "radius": radius,
            "most_cluster": most_cluster,
            "most_cluster_proportion": most_cluster_proportion
        })

# DataFrame 생성 및 반지름 기준으로 정렬
chain_info_df = pd.DataFrame(chain_info).sort_values(by="radius", ascending=False).reset_index(drop=True)

# 테이블 출력
st.write("Chain Information Sorted by Radius")
st.dataframe(chain_info_df)


# Voronoi 다이어그램 생성
voronoi_points = np.array(list(chain_centers.values()))
vor = Voronoi(voronoi_points)

# Voronoi 다이어그램 시각화
fig, ax = plt.subplots(figsize=(10, 10))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=1.5)


cmap = cm.get_cmap("Spectral", 12)
# 반지름을 표시하는 원 추가
for i, (chain, (center_x, center_y)) in enumerate(chain_centers.items()):
    # most_cluster의 색상으로 동그라미 그리기
    color = cmap(most_clusters[i] / 12)  # Normalize cluster number to [0, 1]
    ax.scatter(center_x, center_y, marker='o', s=120, color=color, edgecolor='black')  # 크기 조정
    # 체인 이름을 원 아래에 표시
    ax.text(center_x, center_y - radii[i] * 0.05, chain, fontsize=9, ha='center', va='top', color='black', fontweight='bold')

# 시각화 설정 및 출력
ax.set_title("Voronoi Diagram of Chain Centers with Radius")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
st.pyplot(fig)
import streamlit as st
import random
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ---------- Page Config ----------
st.set_page_config(
    page_title="ShopSort Analytics Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    .scenario-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(102, 126, 234, 0.6);
    }
    
    .info-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    
    .algorithm-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .badge-bubble { background: linear-gradient(135deg, #FF6B6B, #EE5A6F); color: white; }
    .badge-merge { background: linear-gradient(135deg, #4ECDC4, #44A08D); color: white; }
    .badge-quick { background: linear-gradient(135deg, #45B7D1, #2E86AB); color: white; }
    .badge-heap { background: linear-gradient(135deg, #FFA07A, #FA8072); color: white; }
    .badge-radix { background: linear-gradient(135deg, #98D8C8, #7CB9A8); color: white; }
</style>
""", unsafe_allow_html=True)

# ---------- Sorting Algorithms ----------

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1): 
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        merge_sort(left)
        merge_sort(right)
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)

def heapify(arr, n, i):
    largest = i
    l = 2*i + 1
    r = 2*i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

def counting_sort(arr, exp):
    n = len(arr)
    output = [0]*n
    count = [0]*10
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    i = n - 1
    while i >= 0:
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        i -= 1
    for i in range(len(arr)):
        arr[i] = output[i]

def radix_sort(arr):
    if len(arr) == 0:
        return
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort(arr, exp)
        exp *= 10

# ---------- Helper Function ----------

def measure_time(sort_func, arr):
    start = time.time()
    if sort_func == quick_sort:
        result = sort_func(arr.copy())
    else:
        arr_copy = arr.copy()
        sort_func(arr_copy)
    end = time.time()
    return end - start

# ---------- E-Commerce Scenarios ----------

ecommerce_scenarios = {
    "📦 Product Prices": {
        "description": "Sorting products by price for customer browsing experience",
        "data_type": "Random",
        "range": (10, 10000),
        "icon": "💰",
        "color": "#667eea"
    },
    "⭐ Customer Ratings": {
        "description": "Organizing products by customer ratings (1-5 stars)",
        "data_type": "Random",
        "range": (1, 5),
        "icon": "🌟",
        "color": "#f093fb"
    },
    "📊 Order IDs (Sequential)": {
        "description": "Processing orders that arrive in sequential order",
        "data_type": "Sorted",
        "range": (1000, 9999),
        "icon": "📋",
        "color": "#4facfe"
    },
    "🔄 Return Priority": {
        "description": "Processing returns by urgency level (high to low priority)",
        "data_type": "Reverse",
        "range": (1, 100),
        "icon": "⚡",
        "color": "#fa709a"
    },
    "📈 Sales Volume": {
        "description": "Analyzing sales data across different products",
        "data_type": "Random",
        "range": (0, 50000),
        "icon": "📊",
        "color": "#45B7D1"
    }
}

algo_info = {
    'Bubble Sort': {'complexity': 'O(n²)', 'use_case': 'Small datasets', 'color': '#FF6B6B'},
    'Merge Sort': {'complexity': 'O(n log n)', 'use_case': 'Large datasets', 'color': '#4ECDC4'},
    'Quick Sort': {'complexity': 'O(n log n)', 'use_case': 'Average cases', 'color': '#45B7D1'},
    'Heap Sort': {'complexity': 'O(n log n)', 'use_case': 'Priority queues', 'color': '#FFA07A'},
    'Radix Sort': {'complexity': 'O(nk)', 'use_case': 'Integer sorting', 'color': '#98D8C8'}
}

# ---------- Header ----------
st.markdown("""
<div class="main-header">
    <h1>🛒 ShopSort Analytics Pro</h1>
    <p>Advanced E-Commerce Sorting Performance Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/online-store.png", width=80)
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    scenario = st.selectbox(
        "🎯 Select E-Commerce Scenario",
        list(ecommerce_scenarios.keys()),
        format_func=lambda x: f"{ecommerce_scenarios[x]['icon']} {x[2:]}"
    )
    
    st.markdown(f"""
    <div class="info-box">
        <strong>📝 Scenario:</strong><br/>
        {ecommerce_scenarios[scenario]['description']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📏 Dataset Configuration")
    data_sizes = st.multiselect(
        "Dataset Sizes (records)",
        [100, 300, 500, 700, 1000, 1500, 2000, 2500],
        default=[100, 500, 1000, 1500, 2000]
    )
    
    st.markdown("### 🔧 Algorithm Selection")
    algorithms = st.multiselect(
        "Choose Algorithms to Compare",
        ['Bubble Sort', 'Merge Sort', 'Quick Sort', 'Heap Sort', 'Radix Sort'],
        default=['Merge Sort', 'Quick Sort', 'Heap Sort', 'Radix Sort']
    )
    
    st.markdown("---")
    
    run_button = st.button("🚀 RUN ANALYSIS", type="primary")
    
    st.markdown("---")
    st.markdown("### 📚 Quick Reference")
    with st.expander("Algorithm Complexities"):
        for algo, info in algo_info.items():
            st.markdown(f"**{algo}**: `{info['complexity']}`")
    
    with st.expander("Best Use Cases"):
        for algo, info in algo_info.items():
            st.markdown(f"**{algo}**: {info['use_case']}")

# ---------- Main Content ----------

# Algorithm mapping
algo_map = {
    'Bubble Sort': bubble_sort,
    'Merge Sort': merge_sort,
    'Quick Sort': quick_sort,
    'Heap Sort': heap_sort,
    'Radix Sort': radix_sort
}

if run_button:
    if not data_sizes or not algorithms:
        st.error("⚠️ Please select at least one dataset size and one algorithm!")
    else:
        # Analysis Section
        st.markdown("## 📊 Performance Analysis Results")
        
        with st.spinner("🔄 Running comprehensive performance analysis..."):
            data_type = ecommerce_scenarios[scenario]['data_type']
            data_range = ecommerce_scenarios[scenario]['range']
            
            results = {name: [] for name in algorithms}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_iterations = len(data_sizes) * len(algorithms)
            current_iteration = 0
            
            for size in data_sizes:
                base_data = [random.randint(data_range[0], data_range[1]) for _ in range(size)]
                
                for algo_name in algorithms:
                    status_text.text(f"Testing {algo_name} with {size} records...")
                    func = algo_map[algo_name]
                    
                    if data_type == "Random":
                        arr = base_data.copy()
                    elif data_type == "Sorted":
                        arr = sorted(base_data)
                    elif data_type == "Reverse":
                        arr = sorted(base_data, reverse=True)
                    
                    exec_time = measure_time(func, arr)
                    results[algo_name].append(exec_time)
                    
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_iterations)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Analysis Complete! Here are your results:")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Chart", "📊 Data Table", "🏆 Rankings", "💡 Insights"])
            
            with tab1:
                # Interactive Performance Chart
                fig = go.Figure()
                
                for algo_name in algorithms:
                    fig.add_trace(go.Scatter(
                        x=data_sizes,
                        y=results[algo_name],
                        mode='lines+markers',
                        name=algo_name,
                        line=dict(width=4, color=algo_info[algo_name]['color']),
                        marker=dict(size=12, symbol='circle'),
                        hovertemplate=f'<b>{algo_name}</b><br>Size: %{{x}}<br>Time: %{{y:.6f}}s<extra></extra>'
                    ))
                
                fig.update_layout(
                    title={
                        'text': f"<b>Performance Comparison: {scenario}</b>",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 24}
                    },
                    xaxis_title="<b>Dataset Size (records)</b>",
                    yaxis_title="<b>Execution Time (seconds)</b>",
                    hovermode='x unified',
                    height=600,
                    template="plotly_white",
                    font=dict(size=14),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("### 📋 Detailed Performance Metrics")
                df_results = pd.DataFrame(results, index=data_sizes)
                df_results.index.name = "Dataset Size"
                
                # Add statistics
                df_results.loc['Average'] = df_results.mean()
                df_results.loc['Min'] = df_results.iloc[:-1].min()
                df_results.loc['Max'] = df_results.iloc[:-2].max()
                
                st.dataframe(
                    df_results.style.format("{:.6f}")
                    .background_gradient(cmap='RdYlGn_r', axis=1)
                    .highlight_min(axis=1, color='lightgreen')
                    .highlight_max(axis=1, color='lightcoral'),
                    use_container_width=True
                )
                
                # Download button
                csv = df_results.to_csv()
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"sorting_analysis_{scenario.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            
            with tab3:
                st.markdown("### 🏆 Algorithm Rankings")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fastest_small = min(results, key=lambda x: results[x][0])
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h2>🥇</h2>
                        <h3>Fastest for Small Data</h3>
                        <h2>{fastest_small}</h2>
                        <p>{results[fastest_small][0]:.6f}s</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    fastest_large = min(results, key=lambda x: results[x][-1])
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h2>🥇</h2>
                        <h3>Fastest for Large Data</h3>
                        <h2>{fastest_large}</h2>
                        <p>{results[fastest_large][-1]:.6f}s</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_times = {name: sum(times)/len(times) for name, times in results.items()}
                    best_overall = min(avg_times, key=avg_times.get)
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h2>🏆</h2>
                        <h3>Best Overall Average</h3>
                        <h2>{best_overall}</h2>
                        <p>{avg_times[best_overall]:.6f}s</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Speedup Comparison
                st.markdown("### ⚡ Speedup Analysis")
                
                baseline = max(avg_times.values())
                speedup_data = {algo: baseline / avg_time for algo, avg_time in avg_times.items()}
                
                fig_speedup = go.Figure(data=[
                    go.Bar(
                        x=list(speedup_data.keys()),
                        y=list(speedup_data.values()),
                        marker_color=[algo_info[algo]['color'] for algo in speedup_data.keys()],
                        text=[f"{v:.2f}x" for v in speedup_data.values()],
                        textposition='outside'
                    )
                ])
                
                fig_speedup.update_layout(
                    title="Speedup Relative to Slowest Algorithm",
                    xaxis_title="Algorithm",
                    yaxis_title="Speedup Factor",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_speedup, use_container_width=True)
            
            with tab4:
                st.markdown("### 💡 Performance Insights & Recommendations")
                
                # Generate insights
                avg_times = {name: sum(times)/len(times) for name, times in results.items()}
                best_algo = min(avg_times, key=avg_times.get)
                worst_algo = max(avg_times, key=avg_times.get)
                
                st.info(f"""
                **📊 Analysis Summary for {scenario}**
                
                - **Data Pattern**: {data_type}
                - **Dataset Sizes Tested**: {len(data_sizes)} different sizes
                - **Algorithms Compared**: {len(algorithms)} algorithms
                - **Total Tests Performed**: {total_iterations}
                """)
                
                st.success(f"""
                **✅ Best Performer**: **{best_algo}**
                - Average execution time: {avg_times[best_algo]:.6f} seconds
                - Time complexity: {algo_info[best_algo]['complexity']}
                - Ideal for: {algo_info[best_algo]['use_case']}
                """)
                
                st.warning(f"""
                **⚠️ Slowest Performer**: **{worst_algo}**
                - Average execution time: {avg_times[worst_algo]:.6f} seconds
                - Performance ratio: {avg_times[worst_algo]/avg_times[best_algo]:.2f}x slower than best
                """)
                
                # Recommendations
                st.markdown("### 🎯 Recommendations")
                
                if data_type == "Random":
                    st.markdown("""
                    - **Quick Sort** or **Merge Sort** are excellent choices for random data
                    - Avoid **Bubble Sort** for datasets larger than 1000 records
                    - **Radix Sort** performs well with integer data
                    """)
                elif data_type == "Sorted":
                    st.markdown("""
                    - Pre-sorted data benefits from adaptive algorithms
                    - **Merge Sort** maintains consistent performance
                    - **Bubble Sort** can be O(n) for already sorted data
                    """)
                elif data_type == "Reverse":
                    st.markdown("""
                    - Reverse-sorted data is worst-case for some algorithms
                    - **Merge Sort** and **Heap Sort** remain stable
                    - Avoid **Quick Sort** with poor pivot selection
                    """)

else:
    # Landing Page
    st.markdown("## 🎯 Welcome to ShopSort Analytics Pro")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Why Algorithm Performance Matters in E-Commerce
        
        In the fast-paced world of online retail, every millisecond counts. Efficient sorting algorithms 
        can dramatically improve:
        
        - 🚀 **Page Load Speed**: Faster product listings lead to better user experience
        - 💰 **Conversion Rates**: Quick searches increase purchase likelihood
        - 📊 **Data Processing**: Handle millions of products and orders efficiently
        - ⭐ **Customer Satisfaction**: Instant results keep customers happy
        
        ### How to Use This Tool
        
        1. Select an e-commerce scenario from the sidebar
        2. Choose dataset sizes to test
        3. Pick algorithms to compare
        4. Click "RUN ANALYSIS" to see results
        5. Explore insights in different tabs
        """)
    
    with col2:
        st.markdown("### 🔥 Featured Scenarios")
        for scenario_name, info in list(ecommerce_scenarios.items())[:3]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {info['color']}, {info['color']}dd); 
                        padding: 1rem; border-radius: 10px; margin: 0.5rem 0; color: white;">
                <strong>{info['icon']} {scenario_name[2:]}</strong><br/>
                <small>{info['description']}</small>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>ShopSort Analytics Pro</strong> | Powered by Advanced Algorithm Analysis</p>
    <p>Built with ❤️ using Streamlit | © 2024</p>
</div>
""", unsafe_allow_html=True)
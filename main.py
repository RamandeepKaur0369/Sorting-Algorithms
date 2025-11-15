import streamlit as st
import random
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Page Config ----------
st.set_page_config(
    page_title="ShopSort Analytics",
    page_icon="🛒",
    layout="wide"
)

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

# ---------- E-Commerce Context ----------

ecommerce_scenarios = {
    "📦 Product Prices": {
        "description": "Sorting products by price for customer browsing",
        "data_type": "Random",
        "range": (10, 10000)
    },
    "⭐ Customer Ratings": {
        "description": "Organizing products by rating (1-5 stars)",
        "data_type": "Random",
        "range": (1, 5)
    },
    "📊 Order IDs (Sequential)": {
        "description": "Processing orders that arrive in sequence",
        "data_type": "Sorted",
        "range": (1000, 9999)
    },
    "🔄 Return Priority (Reverse)": {
        "description": "Processing returns by urgency (high to low)",
        "data_type": "Reverse",
        "range": (1, 100)
    },
    "📈 Sales Volume": {
        "description": "Analyzing sales data across products",
        "data_type": "Random",
        "range": (0, 50000)
    }
}

# ---------- Streamlit UI ----------

# Header
st.title("🛒 ShopSort Analytics")
st.markdown("### E-Commerce Sorting Algorithm Performance Dashboard")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

scenario = st.sidebar.selectbox(
    "Select E-Commerce Scenario",
    list(ecommerce_scenarios.keys())
)

st.sidebar.info(f"**Scenario:** {ecommerce_scenarios[scenario]['description']}")

data_sizes = st.sidebar.multiselect(
    "Dataset Sizes (records)",
    [100, 300, 500, 700, 1000, 1500, 2000],
    default=[100, 300, 500, 700, 1000]
)

algorithms = st.sidebar.multiselect(
    "Select Algorithms",
    ['Bubble Sort', 'Merge Sort', 'Quick Sort', 'Heap Sort', 'Radix Sort'],
    default=['Bubble Sort', 'Merge Sort', 'Quick Sort', 'Heap Sort', 'Radix Sort']
)

# Algorithm mapping
algo_map = {
    'Bubble Sort': bubble_sort,
    'Merge Sort': merge_sort,
    'Quick Sort': quick_sort,
    'Heap Sort': heap_sort,
    'Radix Sort': radix_sort
}

# Run Analysis Button
if st.sidebar.button("🚀 Run Performance Analysis", type="primary"):
    if not data_sizes or not algorithms:
        st.error("Please select at least one dataset size and one algorithm!")
    else:
        with st.spinner("Analyzing sorting performance..."):
            # Generate data based on scenario
            data_type = ecommerce_scenarios[scenario]['data_type']
            data_range = ecommerce_scenarios[scenario]['range']
            
            results = {name: [] for name in algorithms}
            
            progress_bar = st.progress(0)
            total_iterations = len(data_sizes) * len(algorithms)
            current_iteration = 0
            
            for size in data_sizes:
                base_data = [random.randint(data_range[0], data_range[1]) for _ in range(size)]
                
                for algo_name in algorithms:
                    func = algo_map[algo_name]
                    
                    # Generate appropriate data type
                    if data_type == "Random":
                        arr = base_data.copy()
                    elif data_type == "Sorted":
                        arr = sorted(base_data)
                    elif data_type == "Reverse":
                        arr = sorted(base_data, reverse=True)
                    
                    # Measure time
                    exec_time = measure_time(func, arr)
                    results[algo_name].append(exec_time)
                    
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_iterations)
            
            progress_bar.empty()
            
            # Display Results
            st.success("✅ Analysis Complete!")
            
            # Create interactive plot
            fig = go.Figure()
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            
            for idx, algo_name in enumerate(algorithms):
                fig.add_trace(go.Scatter(
                    x=data_sizes,
                    y=results[algo_name],
                    mode='lines+markers',
                    name=algo_name,
                    line=dict(width=3, color=colors[idx % len(colors)]),
                    marker=dict(size=10)
                ))
            
            fig.update_layout(
                title=f"Performance Comparison: {scenario}",
                xaxis_title="Dataset Size (records)",
                yaxis_title="Execution Time (seconds)",
                hovermode='x unified',
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance Table
            st.subheader("📊 Detailed Performance Metrics")
            
            df_results = pd.DataFrame(results, index=data_sizes)
            df_results.index.name = "Dataset Size"
            
            # Format to 6 decimal places
            st.dataframe(df_results.style.format("{:.6f}"), use_container_width=True)
            
            # Best Algorithm Recommendation
            st.subheader("🏆 Recommendations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Fastest for Small Data",
                    min(results, key=lambda x: results[x][0]) if results else "N/A"
                )
            
            with col2:
                st.metric(
                    "Fastest for Large Data",
                    min(results, key=lambda x: results[x][-1]) if results else "N/A"
                )
            
            with col3:
                avg_times = {name: sum(times)/len(times) for name, times in results.items()}
                st.metric(
                    "Best Overall Average",
                    min(avg_times, key=avg_times.get) if avg_times else "N/A"
                )

# Information Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Algorithm Info")

with st.sidebar.expander("Time Complexities"):
    st.markdown("""
    - **Bubble Sort**: O(n²)
    - **Merge Sort**: O(n log n)
    - **Quick Sort**: O(n log n) avg
    - **Heap Sort**: O(n log n)
    - **Radix Sort**: O(nk)
    """)

# Footer
st.markdown("---")
st.markdown("**ShopSort Analytics** | Optimizing E-Commerce Data Operations | Built with Streamlit")
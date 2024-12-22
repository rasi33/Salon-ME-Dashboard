import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet

# Function to generate mock data (replace with real data in production)
def generate_mock_data():
    customers = pd.DataFrame({
        "Customer ID": range(1, 101),
        "Name": [f"Customer {i}" for i in range(1, 101)],
        "Preferred Service": np.random.choice(["Haircut", "Facial", "Makeup", "Hair Color"], 100),
        "Booking Count": np.random.randint(1, 20, 100),
        "Feedback Score": np.random.randint(1, 5, 100)
    })

    bookings = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=200),
        "Bookings": np.random.randint(5, 50, 200)
    })

    inventory = pd.DataFrame({
        "Product": ["Shampoo", "Conditioner", "Hair Color", "Facial Cream", "Nail Polish"],
        "Stock": [20, 15, 5, 12, 18],
        "Usage (per week)": [5, 4, 3, 6, 7]
    })

    staff = pd.DataFrame({
        "Staff Name": [f"Stylist {i}" for i in range(1, 6)],
        "Clients Served": np.random.randint(50, 200, 5),
        "Revenue Generated": np.random.randint(10000, 50000, 5),
        "Feedback Avg": np.random.randint(3, 5, 5)
    })

    return customers, bookings, inventory, staff

# Generate mock data
customers, bookings, inventory, staff = generate_mock_data()

# Set up the Streamlit app
st.set_page_config(page_title="Interactive Salon ME Dashboard", layout="wide")

# Title and Logo
col1, col2 = st.columns([1, 5])  # Define two columns: one smaller for the logo, and one for the title

# Display Logo in the first column
with col1:
    st.image("./Screenshot_2024-12-21_115932-removebg-preview (1).png", width=250)  # Replace with the path or URL to your logo

# Display Title in the second column
with col2:
    st.title("Salon ME Interactive Data Science Dashboard")

st.sidebar.title("Dashboard Navigation")
options = st.sidebar.radio("Select a view:", ["Overview", "Customer Analytics", "Demand Forecasting", "Inventory Management", "Staff Performance", "Marketing Insights"])

# Service filter for customers
selected_service = st.sidebar.selectbox("Select a Service to Filter Customers", ["All", "Haircut", "Facial", "Makeup", "Hair Color"])

# Overview Section
if options == "Overview":
    st.header("Welcome to the Salon ME Dashboard")
    st.write("""
    This interactive dashboard provides key insights into customer behavior, demand forecasting, inventory levels, staff performance, and marketing effectiveness. Navigate through the sections to explore actionable data insights.
    """)

    # Key metrics with Card Design
    col1, col2, col3, col4 = st.columns(4)

    # Using markdown and CSS to design cards for key metrics
    col1.markdown(
        f"""
        <div style="background-color:#d15115; padding:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1); text-align:center;">
            <h3>Total Customers</h3>
            <p style="font-size:24px; font-weight:bold;">{len(customers)}</p>
        </div>
        """, unsafe_allow_html=True)

    col2.markdown(
        f"""
        <div style="background-color:#d15115; padding:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1); text-align:center;">
            <h3>Total Bookings</h3>
            <p style="font-size:24px; font-weight:bold;">{bookings["Bookings"].sum()}</p>
        </div>
        """, unsafe_allow_html=True)

    col3.markdown(
        f"""
        <div style="background-color:#d15115; padding:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1); text-align:center;">
            <h3>Low Stock Items</h3>
            <p style="font-size:24px; font-weight:bold;">{len(inventory[inventory["Stock"] < 10])}</p>
        </div>
        """, unsafe_allow_html=True)

    col4.markdown(
        f"""
        <div style="background-color:#d15115; padding:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1); text-align:center;">
            <h3>Staff Members</h3>
            <p style="font-size:24px; font-weight:bold;">{len(staff)}</p>
        </div>
        """, unsafe_allow_html=True)

    # Booking Trends Over Time (Line Chart)
    st.subheader("Total Bookings Over Time")
    st.write("""
    This chart shows the total number of bookings received over a specified time period. It helps identify trends, seasonal fluctuations, or periods of increased activity.
    """)
    bookings['Date'] = pd.to_datetime(bookings['Date'])  # Ensure Date is in datetime format
    bookings_grouped = bookings.groupby('Date').sum().reset_index()
    fig = px.line(bookings_grouped, x="Date", y="Bookings", title="Bookings Trend Over Time")
    st.plotly_chart(fig)

# Customer Analytics Section
if options == "Customer Analytics":
    st.header("Customer Analytics")

    st.write("""
    Understand customer behavior and preferences. This section helps you analyze booking counts and feedback scores to identify high-value customers and their preferred services.
    """)

    # Filter customers by selected service
    if selected_service != "All":
        filtered_customers = customers[customers["Preferred Service"] == selected_service]
    else:
        filtered_customers = customers

    # Bar chart for Booking Counts
    st.subheader("Booking Counts by Customers")
    st.write("""
    This bar chart shows the number of bookings made by customers. Use it to identify the most frequent customers.
    """)
    fig = px.bar(filtered_customers, x="Name", y="Booking Count", color="Preferred Service", title="Booking Counts by Customers")
    st.plotly_chart(fig)

    # Feedback Distribution
    st.subheader("Feedback Score Distribution")
    st.write("""
    This histogram displays the distribution of feedback scores from customers, helping you understand overall satisfaction levels.
    """)
    fig = px.histogram(filtered_customers, x="Feedback Score", nbins=5, title="Feedback Score Distribution")
    st.plotly_chart(fig)

# Demand Forecasting Section
if options == "Demand Forecasting":
    st.header("Demand Forecasting")
    st.write("""
    This section provides a forecast of customer bookings for the next 30 days based on historical data. Use this to plan staffing and inventory.
    """)

    # Prophet Model
    df = bookings.rename(columns={"Date": "ds", "Bookings": "y"})
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot forecast
    st.subheader("Booking Forecast")
    st.write("""
    The forecasted trend shows the expected number of bookings in the coming days, allowing for better resource allocation.
    """)
    fig = px.line(forecast, x="ds", y="yhat", title="Booking Forecast for the Next 30 Days")
    st.plotly_chart(fig)

# Inventory Management Section
if options == "Inventory Management":
    st.header("Inventory Management")
    st.write("""
    Monitor stock levels and usage trends to ensure you always have the right products available.
    """)

    # Bar chart for stock levels
    st.subheader("Current Stock Levels")
    st.write("""
    This bar chart highlights the current stock levels of inventory items.
    """)
    fig = px.bar(inventory, x="Product", y="Stock", title="Stock Levels by Product")
    st.plotly_chart(fig)

    # Usage trends
    st.subheader("Weekly Product Usage")
    st.write("""
    Analyze the weekly usage of products to anticipate restocking needs.
    """)
    fig = px.bar(inventory, x="Product", y="Usage (per week)", title="Weekly Usage by Product")
    st.plotly_chart(fig)

# Staff Performance Section
if options == "Staff Performance":
    st.header("Staff Performance")
    st.write("""
    Evaluate staff performance based on clients served, revenue generated, and feedback scores.
    """)

    # Bar chart for clients served
    st.subheader("Clients Served by Staff")
    st.write("""
    This bar chart shows the number of clients served by each staff member, helping identify top performers.
    """)
    fig = px.bar(staff, x="Staff Name", y="Clients Served", title="Clients Served by Staff")
    st.plotly_chart(fig)

    # Revenue generated
    st.subheader("Revenue Generated by Staff")
    st.write("""
    This bar chart highlights the revenue contributions of each staff member.
    """)
    fig = px.bar(staff, x="Staff Name", y="Revenue Generated", title="Revenue Generated by Staff")
    st.plotly_chart(fig)

# Marketing Insights Section
if options == "Marketing Insights":
    st.header("Marketing Insights")
    st.write("""
    Explore the effectiveness of your marketing campaigns and customer retention strategies.
    """)

    st.write("This section is under development. Future features will include campaign performance metrics and customer retention analysis.")

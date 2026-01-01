"""
House Price Prediction App - Streamlit Interface
A professional ML application for predicting house prices using Linear Regression
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import config
import utils


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS
# =============================================================================
def load_css():
    st.markdown("""
    <style>
    /* Base styles */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main header */
    .main-header {
        font-size: 2.25rem;
        font-weight: 600;
        color: #111827;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Sub header */
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Price display */
    .price-display {
        background: #2563eb;
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }

    .price-display h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }

    .price-display p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Insight cards */
    .insight-card {
        background: #f9fafb;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #2563eb;
    }

    /* Feature badges */
    .feature-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 0.25rem;
        background: #eff6ff;
        color: #1e40af;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #9ca3af;
        font-size: 0.85rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
    }

    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


load_css()


# =============================================================================
# SESSION STATE
# =============================================================================
if 'predictions' not in st.session_state:
    st.session_state.predictions = utils.load_prediction_history()


# =============================================================================
# HEADER
# =============================================================================
st.markdown(f'<h1 class="main-header">{config.APP_TITLE}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{config.APP_SUBTITLE}</p>', unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_model():
    return utils.load_model_and_scaler()


model, scaler = load_model()
metrics = utils.load_metrics()


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### Settings")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Price Prediction", "Model Information", "Prediction History", "Market Insights"],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("#### About")
    st.info(f"""
    **Version:** {config.APP_VERSION}

    **Model:** {config.MODEL_TYPE}

    **Features:** 12 parameters

    **Training Data:** 545 houses
    """)

    st.markdown("---")

    st.markdown("#### Quick Tips")
    for tip in config.PRICING_TIPS[:3]:
        st.markdown(f"- {tip}")


# =============================================================================
# MAIN CONTENT
# =============================================================================
if page == "Price Prediction":
    st.header("Price Prediction")
    st.markdown("Enter the property details below to get an estimated price.")

    # Input columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Basic Information")

        area = st.number_input(
            config.FEATURE_LABELS['area'],
            min_value=config.AREA_MIN,
            max_value=config.AREA_MAX,
            value=config.DEFAULT_AREA,
            step=config.AREA_STEP,
            help=config.FEATURE_DESCRIPTIONS['area']
        )

        col1_1, col1_2 = st.columns(2)
        with col1_1:
            bedrooms = st.selectbox(
                config.FEATURE_LABELS['bedrooms'],
                config.BEDROOM_OPTIONS,
                index=config.BEDROOM_OPTIONS.index(config.DEFAULT_BEDROOMS),
                help=config.FEATURE_DESCRIPTIONS['bedrooms']
            )
        with col1_2:
            bathrooms = st.selectbox(
                config.FEATURE_LABELS['bathrooms'],
                config.BATHROOM_OPTIONS,
                index=config.BATHROOM_OPTIONS.index(config.DEFAULT_BATHROOMS),
                help=config.FEATURE_DESCRIPTIONS['bathrooms']
            )

        stories = st.selectbox(
            config.FEATURE_LABELS['stories'],
            config.STORIES_OPTIONS,
            index=config.STORIES_OPTIONS.index(config.DEFAULT_STORIES),
            help=config.FEATURE_DESCRIPTIONS['stories']
        )

        parking = st.selectbox(
            config.FEATURE_LABELS['parking'],
            config.PARKING_OPTIONS,
            index=config.PARKING_OPTIONS.index(config.DEFAULT_PARKING),
            help=config.FEATURE_DESCRIPTIONS['parking']
        )

        with col2:
            st.markdown("#### Amenities & Location")

            # Single column
            mainroad = st.selectbox(
                config.FEATURE_LABELS['mainroad'],
                config.YES_NO_OPTIONS,
                index=1,
                help=config.FEATURE_DESCRIPTIONS['mainroad']
            )

            # Row 1: Guest Room | Basement
            gr_col, bs_col = st.columns(2)

            with gr_col:
                guestroom = st.selectbox(
                    config.FEATURE_LABELS['guestroom'],
                    config.YES_NO_OPTIONS,
                    index=1,
                    help=config.FEATURE_DESCRIPTIONS['guestroom']
                )

            with bs_col:
                basement = st.selectbox(
                    config.FEATURE_LABELS['basement'],
                    config.YES_NO_OPTIONS,
                    index=1,
                    help=config.FEATURE_DESCRIPTIONS['basement']
                )

            # Row 2: Hot Water Heating | Air Conditioning
            hw_col, ac_col = st.columns(2)

            with hw_col:
                hotwaterheating = st.selectbox(
                    config.FEATURE_LABELS['hotwaterheating'],
                    config.YES_NO_OPTIONS,
                    index=1,
                    help=config.FEATURE_DESCRIPTIONS['hotwaterheating']
                )

            with ac_col:
                airconditioning = st.selectbox(
                    config.FEATURE_LABELS['airconditioning'],
                    config.YES_NO_OPTIONS,
                    index=0,
                    help=config.FEATURE_DESCRIPTIONS['airconditioning']
                )
                
            # Row 3: Preferred Area | Furnishing Status
            pa_col, fs_col = st.columns(2)

            with pa_col:
                prefarea = st.selectbox(
                    config.FEATURE_LABELS['prefarea'],
                    config.YES_NO_OPTIONS,
                    index=0,
                    help=config.FEATURE_DESCRIPTIONS['prefarea']
                )

            with fs_col:
                furnishingstatus = st.selectbox(
                    config.FEATURE_LABELS['furnishingstatus'],
                    config.FURNISHING_OPTIONS,
                    index=1,
                    help=config.FEATURE_DESCRIPTIONS['furnishingstatus']
                )


    # Predict button
    st.markdown("---")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        predict_button = st.button("Predict Price", type="primary", use_container_width=True)

    if predict_button:
        # Validate inputs
        is_valid, error_msg = utils.validate_inputs(area, bedrooms, bathrooms, stories, parking)

        if not is_valid:
            st.error(f"Error: {error_msg}")
        else:
            # Encode and predict
            input_data = utils.encode_input(
                area, bedrooms, bathrooms, stories, parking,
                mainroad, guestroom, basement, hotwaterheating,
                airconditioning, prefarea, furnishingstatus
            )

            predicted_price = utils.predict_price(model, scaler, input_data)

            # Save prediction
            utils.save_prediction(
                area, bedrooms, bathrooms, stories, parking,
                mainroad, guestroom, basement, hotwaterheating,
                airconditioning, prefarea, furnishingstatus,
                predicted_price
            )

            # Reload history
            st.session_state.predictions = utils.load_prediction_history()

            # Get insights
            insights = utils.get_price_insight(predicted_price, area)

            # Display results
            st.success("Prediction Complete!")

            # Main price display
            col_price1, col_price2, col_price3 = st.columns([1, 2, 1])
            with col_price2:
                st.markdown(f"""
                <div class="price-display">
                    <h2>{utils.format_price(predicted_price)}</h2>
                    <p>Estimated Property Price</p>
                </div>
                """, unsafe_allow_html=True)

            # Metrics row
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            with col_m1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Price Per Sq Ft</div>
                    <div class="metric-value">{:,.0f}</div>
                </div>
                """.format(insights['price_per_sqft']), unsafe_allow_html=True)

            with col_m2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Category</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(insights['category']), unsafe_allow_html=True)

            with col_m3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Market</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(insights['comparison']), unsafe_allow_html=True)

            with col_m4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Affordability</div>
                    <div class="metric-value">{}/100</div>
                </div>
                """.format(insights['affordability_score']), unsafe_allow_html=True)

            # Price comparison chart
            st.markdown("#### Price Comparison")
            chart_data = utils.create_comparison_chart_data(predicted_price)

            fig = go.Figure(data=[
                go.Bar(
                    x=chart_data['Category'],
                    y=chart_data['Price'],
                    marker_color=['#2563eb', '#94a3b8', '#94a3b8', '#94a3b8'],
                    text=[utils.format_price(p) for p in chart_data['Price']],
                    textposition='outside',
                )
            ])

            fig.update_layout(
                yaxis_title="Price (INR)",
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(
                    gridcolor='rgba(0,0,0,0.05)',
                    tickformat=',.0f'
                ),
                xaxis=dict(gridcolor='rgba(0,0,0,0)')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Feature summary
            st.markdown("#### Property Summary")
            feature_summary = f"""
            <div style="background: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;">
            """

            features = [
                (f"{area:,} sq ft", "Area"),
                (f"{bedrooms} Bedrooms", "Rooms"),
                (f"{bathrooms} Bathrooms", "Rooms"),
                (f"{stories} Stories", "Floors"),
                (f"{parking} Parking", "Parking"),
            ]

            for value, category in features:
                feature_summary += f'<span class="feature-badge">{value}</span>'

            if mainroad == "Yes":
                feature_summary += '<span class="feature-badge">Main Road</span>'
            if airconditioning == "Yes":
                feature_summary += '<span class="feature-badge">Air Conditioning</span>'
            if prefarea == "Yes":
                feature_summary += '<span class="feature-badge">Preferred Area</span>'
            if furnishingstatus != "unfurnished":
                feature_summary += f'<span class="feature-badge">{furnishingstatus.title()}</span>'

            feature_summary += "</div>"
            st.markdown(feature_summary, unsafe_allow_html=True)

elif page == "Model Information":
    st.header("Model Information")

    st.markdown(config.MODEL_DESCRIPTION)

    # Display metrics
    if metrics:
        st.markdown(utils.get_model_summary(metrics))

    # Feature importance
    st.markdown("#### Feature Importance")

    importance_df = utils.get_feature_importance()

    fig = px.bar(
        importance_df.head(8),
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        title='Most Important Features for Price Prediction'
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)')
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "Prediction History":
    st.header("Prediction History")

    if st.session_state.predictions.empty:
        st.info("No predictions yet. Make your first prediction to see it here!")
    else:
        st.dataframe(
            st.session_state.predictions.sort_values('timestamp', ascending=False),
            use_container_width=True,
            column_config={
                'timestamp': st.column_config.DatetimeColumn(
                    'Time',
                    format='MMM DD, YYYY - HH:mm'
                ),
                'predicted_price': st.column_config.NumberColumn(
                    'Price',
                    format='INR %,.2f'
                ),
                'price_per_sqft': st.column_config.NumberColumn(
                    'Price/SqFt',
                    format='INR %,.2f'
                )
            }
        )

        # Summary statistics
        st.markdown("#### Summary Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_price = st.session_state.predictions['predicted_price'].mean()
            st.metric("Average Predicted Price", utils.format_price(avg_price))

        with col2:
            max_price = st.session_state.predictions['predicted_price'].max()
            st.metric("Highest Prediction", utils.format_price(max_price))

        with col3:
            count = len(st.session_state.predictions)
            st.metric("Total Predictions", count)

elif page == "Market Insights":
    st.header("Market Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Market Tips")

        for tip in config.PRICING_TIPS:
            st.markdown(f"""
            <div class="insight-card" style="color: #111827;">
                {tip}
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown(f'<div class="footer">{config.FOOTER_TEXT}</div>', unsafe_allow_html=True)

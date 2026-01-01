"""
House Price Prediction App - Gradio Interface
A professional ML application for predicting house prices using Linear Regression
"""

import gradio as gr
import plotly.graph_objects as go

import config
import utils


# =============================================================================
# MODEL LOADING
# =============================================================================
model, scaler = utils.load_model_and_scaler()
metrics = utils.load_metrics()


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_price(
    area: float,
    bedrooms: int,
    bathrooms: int,
    stories: int,
    parking: int,
    mainroad: str,
    guestroom: str,
    basement: str,
    hotwaterheating: str,
    airconditioning: str,
    prefarea: str,
    furnishingstatus: str
) -> tuple:
    """
    Make a house price prediction with detailed insights.

    Returns:
        Tuple of (formatted_price, metrics_dict, comparison_chart, feature_summary)
    """
    # Validate inputs
    is_valid, error_msg = utils.validate_inputs(area, bedrooms, bathrooms, stories, parking)

    if not is_valid:
        return f"Error: {error_msg}", None, None, None

    # Encode inputs
    input_data = utils.encode_input(
        area, bedrooms, bathrooms, stories, parking,
        mainroad, guestroom, basement, hotwaterheating,
        airconditioning, prefarea, furnishingstatus
    )

    # Make prediction
    predicted_price = utils.predict_price(model, scaler, input_data)

    # Save prediction
    utils.save_prediction(
        area, bedrooms, bathrooms, stories, parking,
        mainroad, guestroom, basement, hotwaterheating,
        airconditioning, prefarea, furnishingstatus,
        predicted_price
    )

    # Get insights
    insights = utils.get_price_insight(predicted_price, area)

    # Format price
    formatted_price = utils.format_price(predicted_price)

    # Create metrics dictionary
    metrics_dict = {
        "price_per_sqft": f"{insights['price_per_sqft']:,.2f}",
        "category": insights['category'],
        "market_comparison": insights['comparison'],
        "affordability_score": f"{insights['affordability_score']}/100",
    }

    # Individual metric values for direct output
    price_per_sqft_val = f"{insights['price_per_sqft']:,.2f}"
    category_val = insights['category']
    market_comparison_val = insights['comparison']
    affordability_score_val = f"{insights['affordability_score']}/100"

    # Create comparison chart
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
        title="Price Comparison",
        yaxis_title="Price (INR)",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            tickformat=',.0f'
        )
    )

    # Create feature summary
    feature_list = [
        f"Area: {area:,} sq ft",
        f"Bedrooms: {bedrooms}",
        f"Bathrooms: {bathrooms}",
        f"Stories: {stories}",
        f"Parking: {parking}",
    ]

    if mainroad == "Yes":
        feature_list.append("Main Road")
    if airconditioning == "Yes":
        feature_list.append("Air Conditioning")
    if prefarea == "Yes":
        feature_list.append("Preferred Area")
    if furnishingstatus != "unfurnished":
        feature_list.append(furnishingstatus.title())

    feature_summary = "\n".join(feature_list)

    return formatted_price, price_per_sqft_val, category_val, market_comparison_val, affordability_score_val, fig, feature_summary


# =============================================================================
# HISTORY FUNCTIONS
# =============================================================================
def get_prediction_history() -> str:
    """Get prediction history as a formatted table."""
    history = utils.load_prediction_history()

    if history.empty:
        return "No predictions yet. Make your first prediction!"

    result = "Recent Predictions:\n\n"
    result += "| Time | Area | Beds | Price | Price/SqFt |\n"
    result += "|" + "-" * 70 + "|\n"

    for _, row in history.sort_values('timestamp', ascending=False).head(10).iterrows():
        result += f"| {row['timestamp']} | {row['area']:,.0f} | {int(row['bedrooms'])} | "
        result += f"{utils.format_price(row['predicted_price'])} | "
        result += f"{row['price_per_sqft']:,.0f} |\n"

    return result


# =============================================================================
# CUSTOM CSS
# =============================================================================
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.predict-btn {
    background: #2563eb !important;
    color: white !important;
    font-weight: 500 !important;
    border: none !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    border-radius: 6px !important;
}

.price-display {
    font-size: 42px !important;
    font-weight: 600 !important;
    color: #2563eb !important;
    text-align: center !important;
    padding: 24px !important;
    background: #eff6ff !important;
    border-radius: 8px !important;
    margin: 16px 0 !important;
}

.metric-card {
    background: white !important;
    padding: 16px !important;
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    text-align: center !important;
}

.metric-label {
    font-size: 12px !important;
    color: #6b7280 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 500 !important;
}

.metric-value {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #111827 !important;
}
"""


# =============================================================================
# GRADIO INTERFACE
# =============================================================================
with gr.Blocks(css=custom_css, title=config.APP_TITLE) as demo:

    gr.Markdown(f"""
    # {config.APP_TITLE}
    ## {config.APP_SUBTITLE}

    Version {config.APP_VERSION} | {config.MODEL_TYPE} Model
    """)

    with gr.Tabs():

        # Tab 1: Price Prediction
        with gr.Tab("Price Prediction"):

            gr.Markdown("### Enter House Details")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Basic Information")

                    area = gr.Number(
                        label=config.FEATURE_LABELS['area'],
                        value=config.DEFAULT_AREA,
                        minimum=config.AREA_MIN,
                        maximum=config.AREA_MAX,
                        info=f"Range: {config.AREA_MIN} - {config.AREA_MAX} sq ft"
                    )

                    with gr.Row():
                        bedrooms = gr.Dropdown(
                            choices=config.BEDROOM_OPTIONS,
                            value=config.DEFAULT_BEDROOMS,
                            label=config.FEATURE_LABELS['bedrooms']
                        )
                        bathrooms = gr.Dropdown(
                            choices=config.BATHROOM_OPTIONS,
                            value=config.DEFAULT_BATHROOMS,
                            label=config.FEATURE_LABELS['bathrooms']
                        )

                    with gr.Row():
                        stories = gr.Dropdown(
                            choices=config.STORIES_OPTIONS,
                            value=config.DEFAULT_STORIES,
                            label=config.FEATURE_LABELS['stories']
                        )
                        parking = gr.Dropdown(
                            choices=config.PARKING_OPTIONS,
                            value=config.DEFAULT_PARKING,
                            label=config.FEATURE_LABELS['parking']
                        )

                with gr.Column(scale=1):
                    gr.Markdown("#### Amenities & Location")

                    mainroad = gr.Radio(
                        choices=config.YES_NO_OPTIONS,
                        value="Yes",
                        label=config.FEATURE_LABELS['mainroad']
                    )
                    guestroom = gr.Radio(
                        choices=config.YES_NO_OPTIONS,
                        value="No",
                        label=config.FEATURE_LABELS['guestroom']
                    )
                    basement = gr.Radio(
                        choices=config.YES_NO_OPTIONS,
                        value="No",
                        label=config.FEATURE_LABELS['basement']
                    )
                    hotwaterheating = gr.Radio(
                        choices=config.YES_NO_OPTIONS,
                        value="No",
                        label=config.FEATURE_LABELS['hotwaterheating']
                    )
                    airconditioning = gr.Radio(
                        choices=config.YES_NO_OPTIONS,
                        value="Yes",
                        label=config.FEATURE_LABELS['airconditioning']
                    )
                    prefarea = gr.Radio(
                        choices=config.YES_NO_OPTIONS,
                        value="Yes",
                        label=config.FEATURE_LABELS['prefarea']
                    )
                    furnishingstatus = gr.Radio(
                        choices=config.FURNISHING_OPTIONS,
                        value="semi-furnished",
                        label=config.FEATURE_LABELS['furnishingstatus']
                    )

            # Predict button
            predict_btn = gr.Button("Predict Price", variant="primary", size="lg", elem_classes="predict-btn")

            # Output section
            gr.Markdown("### Prediction Results")

            with gr.Row():
                price_output = gr.Textbox(
                    label="Predicted Price",
                    elem_classes="price-display",
                    interactive=False
                )

            # Metrics row
            with gr.Row():
                with gr.Column():
                    price_per_sqft = gr.Textbox(label="Price per Sq Ft", interactive=False)
                with gr.Column():
                    category = gr.Textbox(label="Category", interactive=False)
                with gr.Column():
                    market_comparison = gr.Textbox(label="Market Comparison", interactive=False)
                with gr.Column():
                    affordability_score = gr.Textbox(label="Affordability Score", interactive=False)

            # Comparison chart
            comparison_chart = gr.Plot(label="Price Comparison")

            # Feature summary
            feature_summary = gr.Textbox(label="Property Summary", lines=6, interactive=False)

            # Connect button
            predict_btn.click(
                fn=predict_price,
                inputs=[
                    area, bedrooms, bathrooms, stories, parking,
                    mainroad, guestroom, basement, hotwaterheating,
                    airconditioning, prefarea, furnishingstatus
                ],
                outputs=[price_output, price_per_sqft, category, market_comparison, affordability_score, comparison_chart, feature_summary]
            )

        # Tab 2: Model Information
        with gr.Tab("Model Information"):

            gr.Markdown(config.MODEL_DESCRIPTION)

            # Display metrics
            if metrics:
                gr.Markdown(f"""
                ### Model Performance Metrics

                | Metric | Value |
                |--------|-------|
                | R² Score | {metrics.get('r2', 0):.4f} |
                | RMSE | {metrics.get('rmse', 0):,.2f} |
                | MAE | {metrics.get('mae', 0):,.2f} |
                | MSE | {metrics.get('mse', 0):,.2f} |

                ### Interpretation

                - **R² Score**: {metrics.get('r2', 0) * 100:.1f}% of the price variation is explained by the model
                - **RMSE**: Average prediction error is approximately {metrics.get('rmse', 0):,.0f}
                - **MAE**: Mean absolute error is {metrics.get('mae', 0):,.0f}
                """)

            # Feature importance table
            gr.Markdown("### Feature Importance")

            importance_df = utils.get_feature_importance()

            importance_table = gr.Dataframe(
                value=importance_df,
                label="Feature Importance Ranking",
                interactive=False
            )

            gr.Markdown("""
            ### Key Insights

            The correlation values indicate how strongly each feature relates to the house price:

            - **Positive correlation**: Feature increases → Price increases
            - **Negative correlation**: Feature increases → Price decreases

            **Top Features:**
            1. **Area (54%)**: Larger houses cost more
            2. **Bathrooms (45%)**: More bathrooms = higher price
            3. **Air Conditioning (44%)**: AC adds significant value
            4. **Stories (42%)**: Multi-story houses are more expensive
            """)

        # Tab 3: Prediction History
        with gr.Tab("Prediction History"):

            gr.Markdown("### Recent Predictions")

            history_text = gr.Textbox(
                value=get_prediction_history(),
                lines=15,
                interactive=False,
                label="Prediction History"
            )

            refresh_btn = gr.Button("Refresh", size="sm")

            refresh_btn.click(
                fn=get_prediction_history,
                outputs=history_text
            )

        # Tab 4: Tips & Insights
        with gr.Tab("Tips & Insights"):

            gr.Markdown("### Market Insights")

            gr.Markdown("""
            #### Price Categories

            | Category | Price Range |
            |----------|-------------|
            | Budget-Friendly | Under 3,000,000 |
            | Mid-Range | 3,000,000 - 5,000,000 |
            | Premium | 5,000,000 - 8,000,000 |
            | Luxury | Above 8,000,000 |
            """)

            gr.Markdown("### Valuation Tips")

            for tip in config.PRICING_TIPS:
                gr.Markdown(f"- {tip}")

            gr.Markdown("""
            ### How to Get the Best Valuation

            1. **Accurate Area Measurement**: Ensure you measure the total built-up area correctly
            2. **Consider All Amenities**: Include all features like AC, parking, etc.
            3. **Location Matters**: Properties on main roads and preferred areas command higher prices
            4. **Furnishing Status**: Furnished properties are valued higher
            5. **Bathroom Count**: Additional bathrooms significantly increase value

            ### Understanding the Model

            The model uses **Linear Regression** with the following characteristics:
            - 545 training samples
            - 12 input features
            - R² score of ~0.62 (explains 62% of price variance)
            - Best for properties within the training data range
            """)

    # Footer
    gr.Markdown(f"""
    ---

    {config.FOOTER_TEXT}

    **Disclaimer**: This is a predictive model for educational purposes.
    Actual market prices may vary based on many factors not included in this model.
    """)


# =============================================================================
# LAUNCH APP
# =============================================================================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

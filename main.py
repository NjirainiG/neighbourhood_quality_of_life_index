import os
import sys
import firebase_admin
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
from typing import List
import requests
import streamlit as st
import pydeck as pdk
import geopandas as gpd
import matplotlib
from shapely.geometry import Point
from langchain_deepseek import ChatDeepSeek
from langchain_experimental.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
import json
import tempfile
import time
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))
from utils.error_handler import GeoAIError
from dotenv import load_dotenv

from firebase_admin import credentials
from firebase_admin import storage

# Load environment variables
load_dotenv()

# Initialize Firebase with error handling
try:
    # Initialize Firebase with explicit name handling
    try:
        app = firebase_admin.get_app('geoai-app')
    except ValueError:
        # Initialize if not exists
        firebase_config = json.loads(st.secrets["FIREBASE_CONFIG_JSON"])
        cred = credentials.Certificate(firebase_config)
        app = firebase_admin.initialize_app(cred, {
            'storageBucket': st.secrets["FIREBASE_CONFIG_JSON_BUCKET"],
            'location': 'US'
        }, name='geoai-app')
except Exception as e:
    st.error(f"Firebase initialization failed: {str(e)}")
    st.stop()

# Get bucket from initialized app
bucket = storage.bucket(app=app)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def fetch_geojson_from_storage(file_path):
    """Fetches a GeoJSON file from Firebase Storage."""
    try:
        blob = bucket.blob(file_path)
        if not blob.exists():
            raise GeoAIError("Requested file not found in storage", 404)
        downloaded_file = blob.download_as_text()
        return downloaded_file
    except Exception as e:
        raise GeoAIError(f"Failed to fetch file: {str(e)}", 500)

def upload_to_storage(file, destination_path=None):
    """Uploads a file to Firebase Storage."""
    try:
        if not destination_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination_path = f"uploads/{timestamp}_{file.name}"

        blob = bucket.blob(destination_path)
        blob.upload_from_file(file)
        return destination_path
    except Exception as e:
        raise GeoAIError(f"Upload failed: {str(e)}", 500)

def process_user_query(agent, query, gdf):
    """Processes user query using the Langchain agent with robust response handling."""
    try:
        # First check if query is about a specific location (like Mathare)
        # Enhanced neighborhood matching with validation
        official_names = gdf['Name'].astype(str).unique()
        normalized_names = {name.lower(): name for name in official_names}
        
        # Check for exact match first with case insensitivity
        location_query = next((name for name in official_names 
                            if name.lower() == query.lower().strip()), None)
        
        if not location_query:
            # Check for partial matches using cleaned names
            cleaned_query = query.lower().replace(" ", "").strip()
            location_query = next((name for name in official_names 
                                  if cleaned_query in name.lower().replace(" ", "")), None)
        
        if not location_query:
            # Use difflib for fuzzy matching
            from difflib import get_close_matches
            matches = get_close_matches(query, official_names, n=3, cutoff=0.7)
            if matches:
                return {
                    "response": f"'{query}' not found. Did you mean: {', '.join(matches)}?",
                    "type": "suggestions",
                    "options": matches
                }
        
        if location_query:
            # Search all rows for the location
            matches = gdf[gdf['Name'].str.lower().str.contains(location_query)]
            if not matches.empty:
                # Return all data for the location
                return {
                    "response": f"Found data for {location_query.capitalize()}:\n{matches.drop(columns='geometry').to_string(index=False)}",
                    "type": "location_data",
                    "data": matches.to_dict()
                }
            else:
                # Explicitly state no data found after thorough search
                return {
                    "response": f"After searching all {len(gdf)} records, no data was found for {location_query.capitalize()}. Available neighborhoods: {', '.join(gdf['Name'].unique())}",
                    "type": "no_data"
                }

        response = agent.run(query)
        print(f"Agent raw response: {response}")  # Debug logging
        
        # Handle different response types
        if isinstance(response, dict):
            return response  # Already structured response
            
        # Try to parse as JSON if it looks like JSON
        if isinstance(response, str):
            try:
                if response.strip().startswith('{') and response.strip().endswith('}'):
                    return json.loads(response.strip())
            except json.JSONDecodeError:
                pass  # Fall through to other handling
                
        # Handle common quality of life query patterns
        if "quality of life" in query.lower() or "nqoli" in query.lower():
            # Extract scores from text response
            import re
            matches = re.findall(r'(\w+)\s*\(([0-9.]+)\)', response)
            if matches:
                return {
                    "response": response,
                    "scores": {name: float(score) for name, score in matches},
                    "type": "quality_of_life"
                }
        
        # Handle spatial queries
        if any(keyword in query.lower() for keyword in ['map', 'location', 'near', 'distance']):
            return {
                "response": response,
                "visualization": "map",
                "data": gdf.to_dict()
            }
            
        # Default response handling
        return {
            "response": response,
            "type": "text"
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to process query",
            "suggestion": "Try rephrasing your question"
        }

def update_map(gdf, response):
    """Updates the map based on the agent's response."""
    if isinstance(response, dict) and "filter" in response:  # Check if response is a dictionary and contains 'filter'
        # Apply filtering based on the 'filter' key in the response
        filter_expression = response["filter"]
        filtered_gdf = gdf.query(filter_expression)  # Apply the filter using GeoPandas
        return filtered_gdf
    return gdf  # Return the original GeoDataFrame if no filter is present or response is not a dictionary

def display_map(display_gdf, calc_gdf):
    """Displays the GeoJSON data on an interactive map using Folium.
    
    Args:
        display_gdf: GeoDataFrame in WGS84 (EPSG:4326) for visualization
        calc_gdf: GeoDataFrame in Web Mercator (EPSG:3857) for calculations
    """
    if not display_gdf.empty:  # Check if the GeoDataFrame is not empty
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Map", "Statistics", "Data"])
        
        with tab1:
            import folium
            from streamlit_folium import st_folium
            
            # Calculate centroids using projected coordinates
            projected_centroids = calc_gdf.geometry.centroid.to_crs(epsg=4326)
            centroid_y = projected_centroids.y.mean()
            centroid_x = projected_centroids.x.mean()
            
            # Create Folium map centered on data
            m = folium.Map(
                location=[centroid_y, centroid_x],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # Get numeric columns for indicator selection
            numeric_cols = display_gdf.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                # Add indicator selection widget
                selected_indicator = st.selectbox(
                    "Select indicator to visualize:",
                    numeric_cols,
                    key='map_indicator'
                )
                
                # Normalize values for color mapping
                min_val = display_gdf[selected_indicator].min()
                max_val = display_gdf[selected_indicator].max()
                norm_values = display_gdf[selected_indicator].apply(
                    lambda x: (x - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                )
                
                # Viridis color scale
                viridis = matplotlib.colormaps['viridis']
                
                # Create choropleth map
                folium.Choropleth(
                    geo_data=display_gdf,
                    name='choropleth',
                    data=display_gdf,
                    columns=['Name', selected_indicator],
                    key_on='feature.properties.Name',
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name=f'{selected_indicator} Score',
                    highlight=True,
                    bins=7,
                    reset=True
                ).add_to(m)

                # Add tooltips
                folium.features.GeoJson(
                    display_gdf,
                    name='Labels',
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['Name', selected_indicator],
                        aliases=['Neighborhood:', f'{selected_indicator}:'],
                        localize=True
                    )
                ).add_to(m)
                
                # Add color scale legend
                from branca.element import Template, MacroElement
                
                template = """
                {% macro html(this, kwargs) %}
                <div style="
                    position: fixed; 
                    bottom: 50px;
                    left: 50px;
                    width: 150px;
                    height: 80px;
                    z-index:9999;
                    font-size:14px;
                    ">
                    <p style="
                        margin:0;
                        padding:5px;
                        background-color: white;
                        "><b>Indicator Values</b></p>
                    <p style="
                        margin:0;
                        padding:5px;
                        background-color: white;
                        ">High: {{this.max}}</p>
                    <p style="
                        margin:0;
                        padding:5px;
                        background-color: white;
                        ">Low: {{this.min}}</p>
                </div>
                {% endmacro %}
                """
                
                macro = MacroElement()
                macro._template = Template(template)
                macro.max = f"{max_val:.2f}"
                macro.min = f"{min_val:.2f}"
                m.get_root().add_child(macro)
            else:
                # Fallback if no numeric columns
                folium.GeoJson(
                    display_gdf,
                    name='geojson',
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['Name'],
                        aliases=['Place:'],
                        localize=True
                    ),
                    style_function=lambda x: {
                        'fillColor': 'red',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    }
                ).add_to(m)
            
            # Display map in Streamlit
            st_folium(m, width=800, height=500)
            
        with tab2:
            # Statistics view
            st.subheader("Spatial Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Features", len(display_gdf))
                st.metric("Area (sq km)", round(calc_gdf.geometry.area.sum() / 10**6, 2))
            with col2:
                # Use projected centroids for accurate coordinates
                projected_centroids = calc_gdf.geometry.centroid.to_crs(epsg=4326)
                st.metric("Centroid Latitude", round(projected_centroids.y.mean(), 4))
                st.metric("Centroid Longitude", round(projected_centroids.x.mean(), 4))
            
            # Numeric column analysis
            numeric_cols = display_gdf.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Analyze numeric column:", numeric_cols)
                st.bar_chart(display_gdf[selected_col].value_counts())
                
        with tab3:
            # Complete data view with all columns
            st.subheader("Full Dataset")
            st.write(f"Showing all {len(display_gdf)} records")
            st.dataframe(
                display_gdf,
                column_config={
                    "geometry": st.column_config.TextColumn("Geometry (WKT)"),
                },
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "Download Data",
                data=display_gdf.to_csv(index=False),
                file_name="nqoli_data.csv",
                mime="text/csv"
            )
            
    else:
        st.warning("No data to display on the map after filtering.")


def main():
    st.title("GeoAI Neighborhood Quality of Life Explorer-mtaawetu2025")

    # Initialize session state
    if 'gdf' not in st.session_state:
        st.session_state.gdf = None
    if 'active_file' not in st.session_state:
        st.session_state.active_file = None

    # Always use default file
    if st.session_state.gdf is None:
        try:
            # Load complete dataset with all 108 neighborhoods
            geojson_data = fetch_geojson_from_storage('NQoLI_wgs84.geojson') 
            # Read GeoJSON and ensure it has valid geometry
            st.session_state.gdf = gpd.read_file(geojson_data)
            if 'geometry' not in st.session_state.gdf.columns:
                st.session_state.gdf['geometry'] = None
            st.session_state.active_file = 'NQoLI_wgs84.geojson'
        except GeoAIError as e:
            st.warning("Default dataset not found in storage. Please upload a GeoJSON file.")
            st.session_state.gdf = gpd.GeoDataFrame()  # Create empty GeoDataFrame
            return

    # Display initial map
    gdf = st.session_state.gdf
    if not gdf.empty and 'geometry' in gdf.columns and not gdf.geometry.is_empty.all() and not gdf.geometry.isna().all():
        try:
            # Ensure we have geographic coordinates (WGS84)
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            
            # Create projected copies for accurate calculations
            projected_gdf = gdf.to_crs(epsg=3857)  # Web Mercator for calculations
            display_gdf = gdf.copy()  # Keep original for display
            
            # Calculate centroids using projected coordinates
            centroids = projected_gdf.geometry.centroid.to_crs(epsg=4326)
            
            display_map(display_gdf, projected_gdf)
        except Exception as e:
            st.error(f"Failed to display map: {str(e)}")
            st.warning("Showing data table instead")
            st.dataframe(gdf.drop(columns='geometry', errors='ignore'))
    elif not gdf.empty:
        st.warning("No valid geometry data found - showing data table")
        st.dataframe(gdf)
    else:
        st.warning("No data available")

    # Enhanced chat interface
    st.subheader("AI Spatial Assistant-sorry for my latency")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the spatial data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Use the complete GeoDataFrame with all 108 neighborhoods
        df_for_agent = gdf.copy()
        
        # Verify we're using the full dataset
        print(f"Using complete dataset with {len(df_for_agent)} neighborhoods for analysis")
        print(f"Columns available: {', '.join(df_for_agent.columns)}")
        
        # Ensure geometry column is included for spatial analysis
        if 'geometry' not in df_for_agent.columns:
            st.error("Missing geometry column in analysis data")
            st.stop()
        
        # Initialize DeepSeek Chat model
        llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            temperature=0.7
        )

        # Create agent directly with the full GeoDataFrame (108 neighborhoods)
        agent = create_pandas_dataframe_agent(
            llm,
            df_for_agent,
            verbose=True,
            max_iterations=15,  # Increased for more complex spatial reasoning
            handle_parsing_errors="Check your output and make sure it conforms to the requirements!",
            allow_dangerous_code=True,
            agent_executor_kwargs={
                "handle_parsing_errors": """
                    Carefully check your output against the format requirements!
                    Consider spatial relationships and dataset-specific columns.
                """,
                "include_df_in_prompt": True  # Ensure full dataframe context is used
            }
        )
        
        # Process query with enhanced spatial analysis pipeline
        with st.chat_message("assistant"):
            with st.spinner("Analyzing spatial data..."):
                try:
                    # Step 1: Verify data access
                    if gdf.empty:
                        response = "No spatial data available for analysis"
                    else:
                        # Step 2: Specialized spatial analysis
                        # Specialized analysis for different indicator types
                        if any(keyword in prompt.lower() for keyword in ['jobs', 'employment']):
                            if 'Jobs' in gdf.columns:
                                max_jobs = gdf['Jobs'].max()
                                top_places = gdf[gdf['Jobs'] == max_jobs][['Name', 'Jobs']]
                                response = f"Thorough analysis of job accessibility:\n"
                                response += f"Highest jobs location(s):\n{top_places.to_string(index=False)}\n"
                                response += f"\nFull distribution:\n{gdf['Jobs'].describe()}\n"
                                response += f"\nNeighborhoods with jobs > 0.7:\n"
                                high_jobs = gdf[gdf['Jobs'] > 0.7][['Name', 'Jobs']].sort_values('Jobs', ascending=False)
                                response += high_jobs.to_string(index=False) if not high_jobs.empty else "None"
                            else:
                                response = "No job data available in this dataset"
                                
                        elif any(keyword in prompt.lower() for keyword in ['health', 'healthcare', 'hospital']):
                            # Debug: Show available columns
                            print(f"Available columns: {gdf.columns.tolist()}")
                            
                            # Find healthcare column (case insensitive)
                            health_cols = [col for col in gdf.columns if 'health' in col.lower()]
                            if health_cols:
                                health_col = health_cols[0]
                                high_health = gdf[gdf[health_col] > 0.7][['Name', health_col]].sort_values(health_col, ascending=False)
                                if not high_health.empty:
                                    response = "Neighborhoods with healthcare score > 0.7:\n"
                                    response += high_health.to_string(index=False) + "\n\n"
                                    response += f"Top healthcare access:\n{gdf[['Name', health_col]].sort_values(health_col, ascending=False).head(5).to_string(index=False)}"
                                else:
                                    response = f"No neighborhoods found with {health_col} > 0.7"
                                response += f"\n\nHealthcare score distribution:\n{gdf[health_col].describe()}"
                            else:
                                response = f"Could not find healthcare column. Available columns: {gdf.columns.tolist()}"
                        
                        # Step 3: General spatial analysis
                        else:
                            # Enhanced reasoning process with progress updates
                            reasoning_steps = [
                                ("🔍 Understanding your question...", "Parsing spatial query"),
                                ("📊 Examining dataset...", f"Analyzing {len(gdf)} features"),
                                ("🧠 Identifying patterns...", "Running spatial analysis"),
                                ("💡 Formulating response...", "Finalizing insights")
                            ]
                            
                            # Display reasoning progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, (step, detail) in enumerate(reasoning_steps):
                                progress = (i + 1) / len(reasoning_steps)
                                status_text.markdown(f"**{step}**  \n*{detail}*")
                                progress_bar.progress(progress)
                                time.sleep(0.3)
                            
                            # Enhanced RAG-style data retrieval
                            data_context = {
                                "summary_stats": gdf.describe().to_dict(),
                                "top_values": {
                                    col: gdf[col].value_counts().head(3).to_dict()
                                    for col in gdf.select_dtypes(include=['number']).columns
                                },
                                "comparisons": {
                                    "highest": {col: gdf.loc[gdf[col].idxmax()].to_dict() for col in gdf.select_dtypes(include=['number']).columns},
                                    "lowest": {col: gdf.loc[gdf[col].idxmin()].to_dict() for col in gdf.select_dtypes(include=['number']).columns}
                                },
                                "spatial_context": {
                                    "crs": str(gdf.crs),
                                    "centroid": list(gdf.geometry.centroid.iloc[0].coords)[0],
                                    "area": gdf.geometry.area.sum(),
                                    "bounds": gdf.total_bounds.tolist()
                                }
                            }

                            # Get raw data from Data tab
                            raw_data = gdf.drop(columns='geometry').to_dict('records')
                            
                            # Conversational analysis prompt
                            # Get dynamic data examples from actual dataset
                            score_column = next((col for col in gdf.columns if 'score' in col.lower()), 'Score')
                            top_areas = gdf.nlargest(3, score_column)[['Name', score_column]]
                            bottom_areas = gdf.nsmallest(3, score_column)[['Name', score_column]]
                            
                            analysis_prompt = f"""
                            Analyze ALL 108 neighborhoods in this spatial dataset and respond to: "{prompt}"

                            Strict Requirements:
                            1. Use COMPLETE dataset - Never sample or subset
                            2. Validate against all 108 neighborhood records
                            3. Cross-reference multiple indicators
                            4. Include spatial relationships between features

                            Dataset Context:
                            - Total neighborhoods: 108 (FULL SET)
                            - Score column: {score_column}
                            - Score range: {gdf[score_column].min():.2f}-{gdf[score_column].max():.2f}
                            - All neighborhoods: {', '.join(gdf['Name'].unique())}

                            Analysis Process:
                            1. Consider all 108 records in every analysis
                            2. Check spatial autocorrelation patterns
                            3. Compare to dataset-wide percentiles
                            4. Use geometric relationships in calculations
                            """
                            
                            raw_response = agent.run({
                                "input": analysis_prompt,
                                "context": {
                                    "full_data_stats": data_context,
                                    "reasoning_steps": [
                                        "1. Identify relevant columns",
                                        "2. Filter/extract required values",
                                        "3. Perform quantitative comparisons",
                                        "4. Spatial pattern recognition",
                                        "5. Formulate insights"
                                    ]
                                }
                            })
                            
                            # Simple, direct response formatting
                            # Dynamic score column handling
                            score_column = next((col for col in gdf.columns if 'score' in col.lower()), None)
                            
                            if score_column:
                                if "highest" in prompt.lower():
                                    max_score = gdf[score_column].max()
                                    top_places = gdf[gdf[score_column] == max_score]['Name'].values
                                    response = f"Highest {score_column} ({max_score:.2f}) found in: {', '.join(top_places)}"
                                elif "lowest" in prompt.lower():
                                    min_score = gdf[score_column].min()
                                    bottom_places = gdf[gdf[score_column] == min_score]['Name'].values
                                    response = f"Lowest {score_column} ({min_score:.2f}) found in: {', '.join(bottom_places)}"
                                elif "higher than" in prompt.lower() or "above" in prompt.lower():
                                    try:
                                        threshold = float(prompt.split("higher than")[1].strip().split()[0]) if "higher than" in prompt.lower() else float(prompt.split("above")[1].strip().split()[0])
                                        filtered = gdf[gdf[score_column] > threshold]
                                        if len(filtered) > 0:
                                            places = ", ".join([f"{row['Name']} ({row[score_column]:.2f})" for _, row in filtered.sort_values(score_column, ascending=False).iterrows()])
                                            response = f"Areas with score above {threshold}: {places}"
                                        else:
                                            response = f"No areas found with score above {threshold}"
                                    except:
                                        response = "Please specify a valid threshold (e.g. 'above 0.3')"
                                else:
                                    response = raw_response
                            else:
                                response = raw_response
                    
                    # Display response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Handle spatial operations
                    if isinstance(response, dict) and "spatial_operation" in response:
                        gdf = update_map(gdf, response)
                        display_map(gdf)
                        
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                finally:
                    pass  # No temporary files to clean up

    # Enhanced instructions and examples
    with st.sidebar:
        st.header("Neighbourhood Quality of Life Index (NQoLI)")
        st.write("""
        **About NQoLI:**
        The NQoLI measures accessibility to key services within 15 minutes walking distance in Nairobi neighborhoods.
        Indicators include:
        - Healthcare facilities
        - Jobs
        - Schools
        - Kindergartens
        - Marketplaces
        - Parks
        - Places of Worship
        - Restaurants
        - Sports Facilities
        - Minutes to CBD (negative impact)
        - Bars (negative impact)
        
        **How to use:**
        - Ask about specific neighborhoods
        - Compare quality of life indicators
        - Visualize spatial patterns
        - © mtaawetu 2025
        """)
        
        st.divider()
        st.subheader("Example Queries-please wait for response")
        st.write("**Basic Analysis:**")
        st.code("- Which neighborhood has highest quality of life?")
        st.code("- Show areas with healthcare score > 0.7")
        
        st.write("**Comparative Analysis:**")
        st.code("- Compare schools and parks between Westlands and Mathare")
        st.code("- Show neighborhoods with best jobs accessibility")
        
        st.write("**Spatial Patterns:**")
        st.code("- Visualize quality of life distribution")
        st.code("- Show areas within 5km of CBD")

if __name__ == "__main__":
    main()

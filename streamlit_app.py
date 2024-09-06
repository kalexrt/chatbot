import os
import requests
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Weather Chatbot")
st.write(
    "This is a simple weather chatbot that uses OpenAI's GPT-3.5 model to generate responses about the weather. "
    "It uses the Open Weather API to retrieve accurate realtime weather data for your city. "
    "If the given location is not a valid city it uses the open street map to get the city name."
)




# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    llm = ChatOpenAI(temperature=0)

    #Define the tool functions"""

    @tool
    def get_weather_data(city:str):
        """
        Calls the weather Api and returns the weather data

        Args:
        city:str

        Returns:
        str
        """
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv('WEATHER_API_KEY')}&units=metric"
        response = requests.get(url)

        return str(response.json())

    @tool
    def get_city_name(location: str) -> str:
        """Calls the Location API and returns the address data
        Args:
            location: str
        Returns:
            str
        """
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
        headers = {
        'User-Agent': 'MyGeocodingApp/1.0 (your-email@example.com)'
    }
        response = requests.get(url, headers=headers)
        if(len(response.json()) > 0):
            return response.json()[0]
        return "City not found"

    #Bind the tool with the llm

    tools = [get_city_name ,get_weather_data]

    llm_with_tools = llm.bind_tools(tools)

    #System Prompt

    prompt = ChatPromptTemplate.from_messages(
        [(
            "system",
            """
            You are a very powerful weather data expert designed to provide users with accurate and up-to-date weather information. Your main functions include:

            1. Call the API: Retrieve weather information using the location provided by the user. Ensure you include parameters for current weather, forecasts, and any relevant alerts.
            2. Display Information: Present all available details from the API response, including:
                Current temperature
                High and low temperatures
                Feels like temperature
                Humidity
                Wind speed
                Sunrise and sunset times
                Any additional relevant weather conditions or alerts
            3. Validate Location: If the user provides an invalid city name, use a tool to find and suggest a valid city name in English.
            Respond in a clear and organized manner to ensure users receive comprehensive and easy-to-understand weather updates. Refer to the examples below:

            # Examples
            ## Example 1: Valid City Name

            User Input: "What's the weather in San Francisco?"

            System Response: "Sure! Here is the current weather in San Francisco:

                Temperature: 65°F
                High/Low: 70°F / 55°F
                Feels Like: 63°F
                Humidity: 75%
                Wind Speed: 8 mph
                Sunrise: 6:45 AM
                Sunset: 7:15 PM
                Conditions: Partly cloudy

            Let me know if you need any additional information!"

            ## Example 2: Invalid City Name

            User Input: "What's the weather in Springfield?"

            System Response: "I found multiple locations with the name 'Springfield.' Could you please specify the state or provide additional details? For example, Springfield, IL or Springfield, MA."

            ## Example 3: Location Not Specified

            User Input: "I need the weather forecast."

            System Response: "Please provide a city name or location so I can retrieve the weather forecast for you. For example, 'New York City' or 'London.'"

            ## Example 4: Weather Alert

            User Input: "Are there any weather alerts for Miami?"

            System Response: "Here is the current weather for Miami:

                Temperature: 82°F
                High/Low: 86°F / 78°F
                Feels Like: 88°F
                Humidity: 85%
                Wind Speed: 12 mph
                Sunrise: 6:30 AM
                Sunset: 7:00 PM
                Conditions: Thunderstorms

            Alert: Severe thunderstorm warning in effect until 8:00 PM. Please take necessary precautions."
            """
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    #Build the Agent

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    
    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.

    #User Input
    if prompt := st.chat_input("Ask me about the weather of any place"):
        result=agent_executor.invoke({"input": prompt})

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write(result['output'])
        st.session_state.messages.append({"role": "assistant", "content": response})

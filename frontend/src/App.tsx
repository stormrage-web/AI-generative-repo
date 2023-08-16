import React from "react";
import "./App.css";
import MainPage from "./pages/MainPage/MainPage";
import { ConfigProvider } from "antd";

const THEME = {
	token: {
		colorPrimary: "#c2c0ff",
		fontFamily: "League Spartan, sans-serif, -apple-system, BlinkMacSystemFont,\n" +
			"  \"Segoe UI\", \"Roboto\", \"Oxygen\", \"Ubuntu\", \"Cantarell\", \"Fira Sans\",\n" +
			"  \"Droid Sans\", \"Helvetica Neue\"",
	},
};

function App() {
	return (
		<ConfigProvider
			theme={THEME}
		>
			<MainPage />
		</ConfigProvider>
	);
}

export default App;

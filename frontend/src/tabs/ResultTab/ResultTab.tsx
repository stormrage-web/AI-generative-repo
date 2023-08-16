import React, { useContext } from "react";

import CardGroup from "../../widgets/CardGroup/CardGroup";
import { CardsContext } from "../../pages/MainPage/MainPage";
import { Button } from "antd";
import axios from "axios";

const ResultTab = () => {
	const [,,resultCards] = useContext(CardsContext);

	const handleDownload = () => {
		axios.get("http://51.250.91.130:5000/download").catch(() => {
			console.log("axios error");
		});
	};

	return (
		<>
			<Button type="primary" htmlType="submit" onClick={handleDownload}>
				Download all items
			</Button>
			<CardGroup items={resultCards} modal/>
		</>
	);
};

export default ResultTab;
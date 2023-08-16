import React, { createContext, Dispatch, SetStateAction, useState } from "react";
import styles from "./MainPage.module.scss";
import { Menu, MenuProps } from "antd";
import { AppstoreOutlined, BarcodeOutlined } from "@ant-design/icons";
import SourceTab from "../../tabs/SourceTab/SourceTab";
import ResultTab from "../../tabs/ResultTab/ResultTab";
import Item from "../../models/Item";

export const CardsContext = createContext<[Item[], (Dispatch<SetStateAction<Item[]>>) | undefined, Item[], (Dispatch<SetStateAction<Item[]>>) | undefined]>([[], undefined, [], undefined]);

const menuItems: MenuProps["items"] = [
	{
		label: "Source",
		key: "src",
		icon: <BarcodeOutlined />,
	},
	{
		label: "Result",
		key: "result",
		icon: <AppstoreOutlined />,
	},
];
const MainPage = () => {
	const [current, setCurrent] = useState("src");
	const [cards, setCards] = useState<Item[]>([]);
	const [resultCards, setResultCards] = useState<Item[]>([]);

	const onClick: MenuProps["onClick"] = (e) => {
		console.log("click ", e);
		setCurrent(e.key);
	};

	return (
		<CardsContext.Provider value={[cards, setCards, resultCards, setResultCards]}>
			<div className={styles.wrapper}>
				<div className={styles.navigation}>
					<Menu
						onClick={onClick}
						selectedKeys={[current]}
						mode="horizontal"
						items={menuItems}
					/>
				</div>
				{current === "src" ? <SourceTab /> : <ResultTab />}
			</div>
		</CardsContext.Provider>
	);
};

export default MainPage;

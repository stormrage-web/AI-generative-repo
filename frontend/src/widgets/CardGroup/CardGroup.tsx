import React from "react";
import styles from "./CardGroup.module.scss";
import Card from "../../entities/Card/Card";
import Item from "../../models/Item";

export interface CardGroupProps {
	items: Item[];
	modal?: boolean;
}

const CardGroup = ({ items, modal }: CardGroupProps) => {
	return (
		<div className={styles.wrapper}>
			{items.map((item) => (
				<Card key={item.id} id={item.id} url={item.url} name={item.name} modal={modal} />
			))}
		</div>
	);
};

export default CardGroup;

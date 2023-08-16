import React, { useContext } from "react";

import UploadForm from "../../widgets/UploadForm/UploadForm";
import CardGroup from "../../widgets/CardGroup/CardGroup";
import { CardsContext } from "../../pages/MainPage/MainPage";

const SourceTab = () => {
	const [cards] = useContext(CardsContext);

	return (
		<>
			<UploadForm />
			<CardGroup items={cards}/>
		</>
	);
};

export default SourceTab;

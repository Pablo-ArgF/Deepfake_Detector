import React from 'react';
import {Text, Flex, Image } from '@chakra-ui/react';
import { useEffect, useState } from 'react';

const Navbar = ({setView}) => {
  const [modelStructureImage, setModelStructureImage] = useState();  
  const [modelGraphsImage, setModelGraphsImage] = useState();  
  const [confussionMatrix, setConfussionMatrix] = useState();
  const [isCNN, setIsCNN] = useState(true);

  useEffect(() => {
    getModelStructureImage();
    getModelGraphsImage();
    getConfussionMatrix();
  }, []);    

  const getModelStructureImage = async () => {
    var path = isCNN ? 'http://156.35.163.188/api/model/structure/cnn' : 'http://156.35.163.188/api/model/structure/rnn';
    const response = await fetch(path)
    setModelStructureImage(response);
  }

  const getModelGraphsImage = async () => {
    var path = isCNN ? 'http://156.35.163.188/api/model/graphs/cnn' : 'http://156.35.163.188/api/model/graphs/rnn';
    const response = await fetch(path)
    setModelGraphsImage(response);
  }

  const getConfussionMatrix = async () => {
    var path = isCNN ? 'http://156.35.163.188/api/model/confussion/matrix/cnn' : 'http://156.35.163.188/api/model/confussion/matrix/rnn';
    const response = await fetch(path)
    setConfussionMatrix(response);
  }


  return (
    <Flex
      direction={'column'}
      wrap="wrap"
      padding="1rem">
      
      {/* image in base64 stored in modelStructureImage variable */}
      <Image width="18em" src={`data:image/png;base64,${modelStructureImage}`} alt="Estructura del modelo usado"/>
      <Image width="18em" src={`data:image/png;base64,${modelGraphsImage}`} alt="Gráficos de entrenamiento del modelo usado"/>
      <Image width="18em" src={`data:image/png;base64,${confussionMatrix}`} alt="Matriz de confusión de tests durante entrenamiento del modelo usado"/>
    
    </Flex>
  );
};

export default Navbar;

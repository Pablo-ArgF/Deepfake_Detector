import React from 'react';
import {Text, Flex, Image, Grid } from '@chakra-ui/react';
import { useEffect, useState } from 'react';

const ModelDetails = ({setView}) => {
  const [rnnModelStructureImage, setRnnModelStructureImage] = useState();  
  const [cnnModelStructureImage, setCnnModelStructureImage] = useState();  
  const [cnnModelGraphsImage, setCnnModelGraphsImage] = useState();  
  const [rnnModelGraphsImage, setRnnModelGraphsImage] = useState();  
  const [cnnConfussionMatrix, setCnnConfussionMatrix] = useState();
  const [rnnConfussionMatrix, setRnnConfussionMatrix] = useState();

  useEffect(() => {
    loadImages();
  }, []);    
  const loadImages = async () => {
    const cnnModelStructure = await fetch('http://156.35.163.188/api/model/structure/cnn');
    setCnnModelStructureImage('data:image/png;base64,'+ await cnnModelStructure.text());
    
    const rnnModelStructure = await fetch('http://156.35.163.188/api/model/structure/rnn');
    setRnnModelStructureImage('data:image/png;base64,'+ await rnnModelStructure.text());
    
    const cnnModelGraphs = await fetch('http://156.35.163.188/api/model/graphs/cnn');
    setCnnModelGraphsImage('data:image/png;base64,'+ await cnnModelGraphs.text());
    
    const rnnModelGraphs = await fetch('http://156.35.163.188/api/model/graphs/rnn');
    setRnnModelGraphsImage('data:image/png;base64,'+ await rnnModelGraphs.text());
    
    const cnnConfusionMatrix = await fetch('http://156.35.163.188/api/model/confussion/matrix/cnn');
    setCnnConfussionMatrix('data:image/png;base64,'+ await cnnConfusionMatrix.text());
    
    const rnnConfusionMatrix = await fetch('http://156.35.163.188/api/model/confussion/matrix/rnn');
    setRnnConfussionMatrix('data:image/png;base64,'+ await rnnConfusionMatrix.text());
  }

  return (
    <Grid templateColumns="repeat(2, 1fr)" gap={6} padding="1rem">
      {/* Titles */}
      <Flex direction="column" align="center" borderRadius="md" p={4}>
        <Text fontSize="1.6em" fontWeight="bold">
          Modelo de predicción por fotogramas (CNN)
        </Text>
      </Flex>
      <Flex direction="column" align="center" borderRadius="md" p={4}>
        <Text fontSize="1.6em" fontWeight="bold">
          Modelo de predicción por secuencias (RNN)
        </Text>
      </Flex>

      {/* Confusion Matrices */}
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">Matriz de confusión CNN</Text>
        <Image width="60%" src={cnnConfussionMatrix} alt="Matriz de confusión del modelo CNN" />
      </Flex>
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">Matriz de confusión RNN</Text>
        <Image width="60%" src={rnnConfussionMatrix} alt="Matriz de confusión del modelo RNN" />
      </Flex>

      {/* Training Graphs */}
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">Gráficos de entrenamiento CNN</Text>
        <Image width="90%" src={cnnModelGraphsImage} alt="Gráficos de entrenamiento del modelo CNN" />
      </Flex>
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">Gráficos de entrenamiento RNN</Text>
        <Image width="90%" src={rnnModelGraphsImage} alt="Gráficos de entrenamiento del modelo RNN" />
      </Flex>

      {/* Model Structures */}
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">Estructura del modelo CNN</Text>
        <Image width="80%" src={cnnModelStructureImage} alt="Estructura del modelo CNN" />
      </Flex>
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">Estructura del modelo RNN</Text>
        <Image width="80%" src={rnnModelStructureImage} alt="Estructura del modelo RNN" />
      </Flex>
    </Grid>
  );
};

export default ModelDetails;

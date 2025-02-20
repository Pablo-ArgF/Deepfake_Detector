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
    const cnnModelStructure = await fetch('/api/model/structure/cnn');
    setCnnModelStructureImage('data:image/png;base64,'+ await cnnModelStructure.text());
    
    const rnnModelStructure = await fetch('/api/model/structure/rnn');
    setRnnModelStructureImage('data:image/png;base64,'+ await rnnModelStructure.text());
    
    const cnnModelGraphs = await fetch('/api/model/graphs/cnn');
    setCnnModelGraphsImage('data:image/png;base64,'+ await cnnModelGraphs.text());
    
    const rnnModelGraphs = await fetch('/api/model/graphs/rnn');
    setRnnModelGraphsImage('data:image/png;base64,'+ await rnnModelGraphs.text());
    
    const cnnConfusionMatrix = await fetch('/api/model/confussion/matrix/cnn');
    setCnnConfussionMatrix('data:image/png;base64,'+ await cnnConfusionMatrix.text());
    
    const rnnConfusionMatrix = await fetch('/api/model/confussion/matrix/rnn');
    setRnnConfussionMatrix('data:image/png;base64,'+ await rnnConfusionMatrix.text());
  }

  return (
    <Grid templateColumns="repeat(2, 1fr)" gap={6} padding="1rem">
      {/* Titles */}
      <Flex direction="column" align="center" borderRadius="md" p={4}>
        <Text fontSize="1.6em" fontWeight="bold">
        Frame-wise prediction model (CNN)
        </Text>
      </Flex>
      <Flex direction="column" align="center" borderRadius="md" p={4}>
        <Text fontSize="1.6em" fontWeight="bold">
        Sequence prediction model (RNN)
        </Text>
      </Flex>

      {/* Confusion Matrices */}
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">CNN confusion matrix</Text>
        <Image width="60%" src={cnnConfussionMatrix} alt="CNN confusion matrix" />
      </Flex>
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">RNN confusion matrix</Text>
        <Image width="60%" src={rnnConfussionMatrix} alt="RNN confusion matrix" />
      </Flex>

      {/* Training Graphs */}
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">CNN training graphs</Text>
        <Image width="90%" src={cnnModelGraphsImage} alt="CNN training graphs" />
      </Flex>
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">RNN training graphs</Text>
        <Image width="90%" src={rnnModelGraphsImage} alt="RNN training graphs" />
      </Flex>

      {/* Model Structures */}
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">CNN model structure</Text>
        <Image width="80%" src={cnnModelStructureImage} alt="CNN model structure" />
      </Flex>
      <Flex direction="column" align="center" bg="grey" borderRadius="0.7em" p={4}>
        <Text textColor={'white'} fontSize="1.3em" fontWeight="bold">RNN model structure</Text>
        <Image width="80%" src={rnnModelStructureImage} alt="RNN model structure" />
      </Flex>
    </Grid>
  );
};

export default ModelDetails;

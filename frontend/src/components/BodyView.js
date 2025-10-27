import React, { useState } from 'react';
import {
  Heading,
  Text,
  FormControl,
  Flex,
  Input,
  InputGroup,
  Button,
  Box,
  Spinner,
  IconButton
} from '@chakra-ui/react';
import { IoMdVideocam } from 'react-icons/io';
import { ArrowBackIcon } from '@chakra-ui/icons';
import CNNVideoDashboard from './CNNVideoDashboard';
import RNNVideoDashboard from './RNNVideoDashboard';

const BodyView = () => {
  const [error, setError] = useState('');
  const [data, setData] = useState(null);
  const [RNNdata, setRNNData] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(1);
  const [videoUploaded, setVideoUploaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [RNNloading, setRNNLoading] = useState(true);
  const [useRNN, setUseRNN] = useState(false);

  const handleVideoUpload = async (event) => {
    setError('');
    const file = event.target.files[0];
    if (!file) {
      setError('Please select a video file.');
      return;
    }
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('video', file);

      const controller = new AbortController();
      var timeoutId = setTimeout(() => controller.abort(), 600000);

      fetch('/api/predict', {
          method: 'POST',
          body: formData,
          headers: {
            'enctype': 'multipart/form-data'
          },
          signal: controller.signal
        }).then(async response => {
          if (!response.ok) {  
              setError('An error occurred while processing the video, please try again.');
              setLoading(false);
              setVideoUploaded(false);
              setData(null);
              return;
          }
          var CNNdata;
          try {
            CNNdata = await response.json();
          } catch (error) {
            setError('An error occurred while processing the video, please try again.');
            setLoading(false);
            setVideoUploaded(false);
            setData(null);
            return;
          }
          setData(CNNdata);
          setLoading(false);
          setVideoUploaded(true);
          clearTimeout(timeoutId);

          var timeoutIdRNN = setTimeout(() => controller.abort(), 600000);
          //TODO Uncomment when CNN is implemented
          // fetch('/api/predict/sequences', {
          //   method: 'POST',
          //   body: formData,
          //   headers: {
          //     'enctype': 'multipart/form-data'
          //   },
          //   signal: controller.signal
          // }).then(async RNNresponse => {
          //       try {
          //         const RNNTmpdata = await RNNresponse.json();
          //         setRNNData(RNNTmpdata);
          //         setRNNLoading(false);
          //       } catch (error) {
          //         setError('An error occurred while processing the video sequences.');
          //         setLoading(false);
          //         setVideoUploaded(false);
          //       }
          //       clearTimeout(timeoutIdRNN);
          //     }).catch(error => {
          //       setError('An error occurred while processing the video sequences.');
          //       setLoading(false);
          //       setVideoUploaded(false);
          //     });
      });
    } catch (error) {
      setError('Error predicting DeepFakes: ' + error);
      setLoading(false);
      setVideoUploaded(false);
      setData(null);
    }
  };

  return (
    <Box>
      <Flex direction='row' width={'100%'} justifyContent={'space-evenly'} visibility={videoUploaded ? 'visible' : 'hidden'}>
      <IconButton
            colorScheme='grey'
            padding={'0.7em'}
            borderRadius={'1em'}
            width={'3em'}
            height={'3em'}
            marginLeft={'1em'}
            marginRight={'1.5em'}
            marginTop={'1em'}
            icon={<ArrowBackIcon boxSize={'2.8em'}/>}
            aria-label='Back to main menu'
            onClick={() => {
              setVideoUploaded(false); 
              setData(null);
              setUseRNN(false);
              setRNNData(null);
              setRNNLoading(true);
              setLoading(false);
            }}
          />
        <Heading w='100%' as="h1" size="5xl" padding={'0.1em'} flexGrow={4}>
          {useRNN ? 'Sequence Analysis (RNN)' : 'Frame-by-Frame Analysis (CNN)' }
        </Heading>
        <Button 
          marginRight={'1em'}
          marginTop={'1em'}
          borderRadius={'1em'}
          backgroundColor={'black'}
          textColor={'white'}
          fontSize={'1em'}
          padding={'0.7em'}
          colorScheme='blue'
          w={'15em'}
          h={'3em'}
          onClick={() => setUseRNN(!useRNN)}
          cursor='pointer'
        >
          {useRNN ? 'Analyze frame by frame' : 'Analyze sequences'}
        </Button>
      </Flex>
      {videoUploaded ? (
        useRNN ? (
          <RNNVideoDashboard 
            setData={setRNNData}
            setLoading={setRNNLoading}
            loading={RNNloading}
            data={RNNdata} 
            setSelectedIndex={setSelectedIndex} 
            selectedIndex={selectedIndex} 
          />
        ) : (
          <CNNVideoDashboard 
            setVideoUploaded={setVideoUploaded} 
            setData={setData}
            setLoading={setLoading}
            data={data} 
            setSelectedIndex={setSelectedIndex} 
            selectedIndex={selectedIndex} 
          />
        )
      ) : (
        <Flex direction="column" align="center" padding="2em">
          <Heading as="h1" size="5xl" mb={15}>
            DeepFake Detection
          </Heading>
          <Text textAlign="justify" width="45%">
            The increasing use of artificial intelligence has enabled identity impersonation through <b>DeepFakes</b>, synthetic content generated by AI algorithms that combine and overlay existing images and videos to create a new one.<br/><br/>
            The realism achieved using current DeepFake algorithms poses a <b>risk to society</b>. From 'fake news' to identity theft, AI-generated content makes it increasingly difficult to distinguish real from fake.<br/><br/>
            This <b>DeepFake detection tool</b> aims to help identify synthetic material through predictive model analysis.
          </Text>
          <Text color="red" marginTop="1em">
            {error}
          </Text>
          <FormControl padding="0.5em">
            <InputGroup marginTop="1em" marginBottom="1em">
              <Input
                type="file"
                id="videoInput"
                accept="video/mp4"
                onInput={handleVideoUpload}
                style={{ display: 'none' }} // Hide the default file input
              />
              <Button as="label" cursor={'pointer'} leftIcon={<IoMdVideocam color='white'/>} htmlFor="videoInput" backgroundColor={'black'} textColor={'white'} padding={'1.1em'} fontSize={'1.1em'}>
                Upload a video
              </Button>
            </InputGroup>
          </FormControl>
          {
            loading ? (
              <Spinner size='10xl' boxSize={'3em'}  thickness='0.2em' colorScheme='blue'/>
            ) : null
          }  
        </Flex>
      )}
    </Box>
  );
};

export default BodyView;

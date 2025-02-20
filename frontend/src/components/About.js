import {
    Heading,
    Text,
    Flex,
    Image
  } from '@chakra-ui/react';
  
  const About = () => {
  
      return(
          <Flex direction="column" align="center" padding="2em">
              <Heading as="h1" size="5xl" mb={15}>
                  Project Information
              </Heading>
              <Flex direction="Row"  placeContent={'center'} borderRadius="md" p={4}>
                  <Flex direction="column" placeContent="center" borderRadius="md" p={4}>
                      <Image h="20em" src="./face.JPEG" alt="Pablo Argallero Fernández" />
                      <Flex direction="row" w={'100%'} alignItems="flex-end" borderRadius="md"  p={4} >
                          <Image  borderRadius='full' w={'3em'} cursor={'pointer'} src='./github.png' onClick={()=>{window.open('https://github.com/Pablo-ArgF', '_blank')}} alt='Access to Pablo-ArgF GitHub'/>
                          <Image  borderRadius='full' w={'3em'} cursor={'pointer'} src='./linkedin.png' onClick={()=>{window.open('https://www.linkedin.com/in/pablo-argallero/', '_blank')}} alt='Access to Pablo-ArgF LinkedIn'/>
                      </Flex>
                  </Flex>
                  <Text  width="30%" textAlign='justify' fontSize={'1.1em'} paddingLeft={'1em'} >
                      <b>Pablo Argallero Fernández</b>, a Computer Engineering student at the University of Oviedo, has developed this project as his Bachelor's Thesis. <br/><br/>
                      The aim of the project is to study alternatives for DeepFake detection in videos, using two deep learning models: Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).<br/><br/>              
                      The creation of this project involved research work, image preprocessing, model training, and result analysis. Additionally, a web interface was implemented to facilitate interaction with the models.<br/><br/>
                      The project has been supervised by professor <b>Cristian González García</b>. 
                  </Text>
              </Flex>
          </Flex>
      )
  }
  
  export default About;
  
// Navbar.js
import React from 'react';
import { Box, Text, Flex, IconButton, useDisclosure } from '@chakra-ui/react';
import { CloseIcon, HamburgerIcon } from '@chakra-ui/icons';

const Navbar = () => {
  const { isOpen, onToggle } = useDisclosure();

  return (
    <Flex
      as="nav"
      align="center"
      justify="space-between"
      wrap="wrap"
      padding="1rem"
      bg="gray.800" // Dark gray background
      color="white"
    >
      {/* Logo */}
      <Box>
        <Text fontSize="lg" fontWeight="bold">
          Logo
        </Text>
      </Box>

      {/* MenuToggle button */}
      <IconButton
        display={{ base: 'block', md: 'none' }}
        onClick={onToggle}
        icon={isOpen ? <CloseIcon /> : <HamburgerIcon />}
        variant="ghost"
        aria-label="Toggle navigation"
      />

      {/* MenuLinks */}
      <Box
        display={{ base: isOpen ? 'block' : 'none', md: 'block' }}
        mt={{ base: 4, md: 0 }}
      >
        <Text mr={4} cursor="pointer">
          About
        </Text>
        <Text cursor="pointer">Model</Text>
      </Box>
    </Flex>
  );
};

export default Navbar;

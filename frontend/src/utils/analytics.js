/**
 * Analytics utility for tracking events with Plausible
 * 
 * This module provides helper functions to track user interactions
 * throughout the Deepfake Detector application.
 */

/**
 * Track a custom event
 * @param {string} eventName - Name of the event to track
 * @param {object} props - Optional properties to attach to the event
 */
export const trackEvent = (eventName, props = {}) => {
    if (window.plausible) {
        window.plausible(eventName, { props });
    } else {
        console.warn('Plausible not loaded, event not tracked:', eventName);
    }
};

/**
 * Track demo access
 */
export const trackDemoAccess = () => {
    trackEvent('Demo Access');
};

/**
 * Track CNN demo access
 */
export const trackCNNDemoAccess = () => {
    trackEvent('CNN Demo Access');
};

/**
 * Track RNN demo access
 */
export const trackRNNDemoAccess = () => {
    trackEvent('RNN Demo Access');
};

/**
 * Track tutorial button click
 * @param {string} modelType - 'cnn' or 'rnn'
 */
export const trackTutorialClick = (modelType) => {
    trackEvent('Tutorial Click', { model: modelType });
};

/**
 * Track video upload submission
 */
export const trackVideoUploadSubmit = () => {
    trackEvent('Video Upload Submit');
};

/**
 * Track successful video analysis
 * @param {string} modelType - 'cnn' or 'rnn'
 */
export const trackVideoAnalysisSuccess = (modelType) => {
    trackEvent('Video Analysis Success', { model: modelType });
};

/**
 * Track failed video analysis
 * @param {string} modelType - 'cnn' or 'rnn'
 * @param {string} errorMessage - Error message
 */
export const trackVideoAnalysisError = (modelType, errorMessage) => {
    trackEvent('Video Analysis Error', {
        model: modelType,
        error: errorMessage
    });
};

/**
 * Track model switch (CNN <-> RNN)
 * @param {string} from - Model switching from
 * @param {string} to - Model switching to
 */
export const trackModelSwitch = (from, to) => {
    trackEvent('Model Switch', { from, to });
};

/**
 * Track navigation to About page
 */
export const trackAboutPageView = () => {
    trackEvent('About Page View');
};

/**
 * Track navigation to Model Details page
 */
export const trackModelDetailsPageView = () => {
    trackEvent('Model Details Page View');
};

/**
 * Track frame selection in CNN dashboard
 * @param {number} frameIndex - Index of the selected frame
 */
export const trackFrameSelection = (frameIndex) => {
    trackEvent('Frame Selection', { frameIndex });
};

/**
 * Track video playback in dashboards
 * @param {string} modelType - 'cnn' or 'rnn'
 */
export const trackVideoPlayback = (modelType) => {
    trackEvent('Video Playback', { model: modelType });
};

import { HistoryItem } from './types';

async function fetchHistory(): Promise<HistoryItem[]> {
  const oneDayAgo = new Date().getTime() - 24 * 60 * 60 * 1000;
  return new Promise((resolve) => {
    chrome.history.search(
      {
        text: '',
        startTime: oneDayAgo,
        maxResults: 1000,
      },
      (results) => {
        const historyItems: HistoryItem[] = results.map((item) => ({
          url: item.url || '',
          title: item.title || '',
          lastVisitTime: item.lastVisitTime || 0,
        }));
        resolve(historyItems);
      }
    );
  });
}

async function sendToBackend(historyItems: HistoryItem[]): Promise<void> {
  try {
    const response = await fetch('http://localhost:8001/api/ingest', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(historyItems),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    console.log('History sent successfully:', await response.json());
  } catch (error) {
    console.error('Error sending history:', error);
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action === 'exportHistory') {
    fetchHistory().then(sendToBackend).then(() => sendResponse({ status: 'success' }));
    return true; // Keep message channel open for async response
  }
});
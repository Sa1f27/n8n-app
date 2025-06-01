document.addEventListener('DOMContentLoaded', () => {
  const button = document.getElementById('exportHistory') as HTMLButtonElement;
  button.addEventListener('click', () => {
    chrome.runtime.sendMessage({ action: 'exportHistory' }, (response) => {
      alert(response.status === 'success' ? 'History exported successfully!' : 'Failed to export history.');
    });
  });
});
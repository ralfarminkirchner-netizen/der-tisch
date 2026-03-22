const CACHE_NAME = 'tisch-v1';
const OFFLINE_URLS = [];

self.addEventListener('install', event => {
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(clients.claim());
});

self.addEventListener('fetch', event => {
  // Pass through to network — apps are server-rendered
  event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
});

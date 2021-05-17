import App from './App.svelte';
import '../public/bare.min.css';

const app = new App({
	target: document.body,
	props: {
		name: 'world'
	}
});

export default app;
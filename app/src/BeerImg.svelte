<script lang="ts">
import { subscribe } from 'svelte/internal';

	let img = "./tmp.png"
	export let beer_data;
	import { actual_beer} from './store.js';
	let act_beer;
	actual_beer.subscribe(value => {
		act_beer=value
	})
	function handleClick() {
		actual_beer.set(beer_data)
	}
	
</script>
	{#if typeof beer_data !=='undefined'}
	<div on:click={handleClick}>
		<img class="beer_img" src={beer_data[3]}>
		<center><div class="beer_desc">Ocena {beer_data[2]}</div></center>
	</div>
	{/if}
<style>
	.beer_img{
		width: 100%;
	}
	.beer_desc{
		position: relative;
		bottom:25px;
		color:#e6a14e;
		background: rgba(0, 0, 0, 0.7);
	}
</style>
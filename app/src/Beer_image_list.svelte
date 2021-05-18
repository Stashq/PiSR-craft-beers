<script lang="ts">
import { each } from 'svelte/internal';


import App from './App.svelte';
import BeerImg from './BeerImg.svelte';
import BeerList from './BeerImg.svelte'
let img = "./tmp.png"
export let beerlist;
</script>
	<div class="beer_img_container">
			{#if typeof beerlist.data !=='undefined'}
				{#each Object.values(beerlist.data) as beer }
					<BeerImg beer_data={beer}></BeerImg>	
				{/each}}
			{/if}
			
		
		


	</div>
	

<style>
	.beer_img_container{
		width: 100%;
		display: flex;
		justify-content: space-around;
	}
</style>
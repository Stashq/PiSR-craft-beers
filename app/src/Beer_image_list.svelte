<script lang="ts">
import { each } from 'svelte/internal';


import App from './App.svelte';
import BeerImg from './BeerImg.svelte';
import BeerList from './BeerImg.svelte'
import Carousel from '@beyonk/svelte-carousel'
import { ChevronLeftIcon, ChevronRightIcon } from 'svelte-feather-icons'
let img = "./tmp.png"
export let beerlist;

</script>
	<div class="beer_img_container">
			{#if typeof beerlist.data !=='undefined'}
			<Carousel loop={true} perPage={4} dots={false} draggable={false}>
				<span class="control" slot="right-control">
					<ChevronRightIcon />
				  </span>
				{#each Object.values(beerlist.data) as beer }
					<BeerImg beer_data={beer}></BeerImg>	
				{/each}}
			</Carousel>
			{/if}
	</div>
	
	

<style>
	.beer_img_container{
		width: 100%;
		display: flex;
		justify-content: space-around;
	}
	.control :global(svg) {
        width: 100%;
        height: 100%;
		margin-left: 55px;
        color: #fff;
        border: 2px solid #fff;
        border-radius: 32px;
    }
</style>
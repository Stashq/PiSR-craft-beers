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
			<Carousel loop={true} perPage={5} dots={false} draggable={false} controls={true}>
				<span class="control_left" slot="left-control">
					<ChevronLeftIcon />
				  </span>
				<span class="control_right" slot="right-control">
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
	.control_right :global(svg) {
        width: 100%;
        height: 100%;
		position:relative;
		left:20px;
        color: #fff;
        border: 2px solid #fff;
        border-radius: 32px;
    }
	.control_left :global(svg) {
        width: 100%;
        height: 100%;
		position:relative;
		right:20px;
        color: #fff;
        border: 2px solid #fff;
        border-radius: 32px;
    }
</style>
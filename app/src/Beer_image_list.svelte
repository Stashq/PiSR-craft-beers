<script lang="ts">

import BeerImg from './BeerImg.svelte';
	import BeerList from './BeerImg.svelte'
	let img = "./tmp.png"
</script>
	<div class="beer_img_container">
		
		
		<BeerImg></BeerImg>
		<BeerImg></BeerImg>
		<BeerImg></BeerImg>
		<BeerImg></BeerImg>


	</div>
	

<style>
	.beer_img_container{
		width: 100%;
		display: flex;
		justify-content: space-around;
	}
</style>